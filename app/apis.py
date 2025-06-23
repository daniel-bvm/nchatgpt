from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from app.oai_models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionMessageParam, ChatCompletionStreamResponse, random_uuid
from app.oai_streaming import create_streaming_response, ChatCompletionResponseBuilder
from app.tools import compose as compose_mcp, get_bio
from app.utils import (
    refine_mcp_response, 
    convert_mcp_tools_to_openai_format, 
    execute_openai_compatible_toolcall,
    refine_chat_history,
    refine_assistant_message,
)
import os
from typing import Optional, Any, AsyncGenerator
from app.configs import settings
import json
import time
import logging

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/v1")

async def get_system_prompt(newest_message: Optional[str]) -> str:
    if not os.path.exists("system_prompt.txt"):
        with open("system_prompt.txt", "w") as f:
            f.write("You are a helpful assistant.")

    with open("system_prompt.txt", "r") as f:
        system_prompt = f.read()

    if newest_message is None:
        return system_prompt

    memory = await get_bio(newest_message)
    memory_str = ""

    for m in memory:
        memory_str += f"- {m}\n"

    if len(memory) > 0:
        logger.info(f"Memory:\n{memory_str}")
        system_prompt += f"\n{system_prompt}\n\nBio:\n{memory_str}"
    
    return system_prompt

def get_newest_message(messages: list[ChatCompletionMessageParam]) -> str:
    if isinstance(messages[-1].get("content", ""), str):
        return messages[-1].get("content", "")
    
    elif isinstance(messages[-1].get("content", []), list):
        for item in messages[-1].get("content", []):
            if item.get("type") == "text":
                return item.get("text", "")

    else:
        raise ValueError(f"Invalid message content: {messages[-1].get('content')}")
    

async def wrap_toolcall_request(uuid: str, fn_name: str, args: dict[str, Any]) -> ChatCompletionStreamResponse:
    args_str = json.dumps(args, indent=2)
    
    template = f'''
<action>Executing <b>{fn_name}</b></action>

<details>
<summary>
Arguments:
</summary>

```json
{args_str}
```

</details>
'''

    return ChatCompletionStreamResponse(
        id=uuid,
        object='chat.completion.chunk',
        created=int(time.time()),
        model='unspecified',
        choices=[
            dict(
                index=0,
                delta=dict(
                    content=template,
                    role='tool'
                ),
            )
        ]
    )
    
async def handle_request(request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionStreamResponse | ChatCompletionResponse, None]:
    messages = request.messages
    assert len(messages) > 0, "No messages in the request"
 
    newest_message = get_newest_message(messages)
    system_prompt = await get_system_prompt(newest_message)
    messages: list[dict[str, Any]] = refine_chat_history(messages, system_prompt)

    tools = await compose_mcp._mcp_list_tools()
    oai_tools = convert_mcp_tools_to_openai_format(tools)
    finished = False
    n_calls, max_calls = 0, 25

    use_tool_calls = lambda: n_calls < max_calls and not finished

    while not finished:
        completion_builder = ChatCompletionResponseBuilder()
    
        payload = dict(
            messages=messages,
            tools=oai_tools,
            tool_choice="auto",
            model=settings.llm_model_id
        )

        if not use_tool_calls():
            payload.pop("tools")
            payload.pop("tool_choice")
        
        logger.info(f"Payload - URL: {settings.llm_base_url}, API Key: {'*' * len(settings.llm_api_key)}, Model: {settings.llm_model_id}")
        streaming_iter = create_streaming_response(
            settings.llm_base_url,
            settings.llm_api_key,
            **payload
        )

        async for chunk in streaming_iter:
            completion_builder.add_chunk(chunk)

            if chunk.choices[0].delta.content:
                yield chunk

        completion = await completion_builder.build()
        messages.append(refine_assistant_message(completion.choices[0].message))

        for call in (completion.choices[0].message.tool_calls or []):
            _id, _name, _args = call.id, call.function.name, call.function.arguments
            _args = json.loads(_args)

            logger.info(f"Executing tool call: {_name} with args: {_args}")
            _result = await execute_openai_compatible_toolcall(_name, _args, compose_mcp)
            logger.info(f"Tool call {_name} result: {_result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _id,
                    "content": refine_mcp_response(_result)
                }
            )

            n_calls += 1

        finished = len((completion.choices[0].message.tool_calls or [])) == 0

    yield completion

@api_router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    enqueued = time.time()
    ttft, tps, n_tokens = float("inf"), None, 0
    req_id = request.request_id or f"req-{random_uuid()}"

    if request.stream:
        generator = handle_request(request)

        async def to_bytes(gen: AsyncGenerator) -> AsyncGenerator[bytes, None]:
            nonlocal ttft, tps, n_tokens

            async for chunk in gen:
                current_time = time.time()

                n_tokens += 1
                ttft = min(ttft, current_time - enqueued)
                tps = n_tokens / (current_time - enqueued)

                if isinstance(chunk, ChatCompletionStreamResponse):
                    data = chunk.model_dump_json()
                    yield "data: " + data + "\n\n"

            logger.info(f"Request {req_id} - TTFT: {ttft:.2f}s, TPS: {tps:.2f} tokens/s")
            yield "data: [DONE]\n\n"

        return StreamingResponse(to_bytes(generator), media_type="text/event-stream")
    
    else:
        async for chunk in handle_request(request):
            current_time = time.time()

            n_tokens += 1
            ttft = min(ttft, current_time - enqueued)
            tps = n_tokens / (current_time - enqueued)

        logger.info(f"Request {req_id} - TTFT: {ttft:.2f}s, TPS: {tps:.2f} tokens/s")
        return JSONResponse(chunk.model_dump())
