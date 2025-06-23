from googlesearch import search, SearchResult
from fastmcp import FastMCP
import ast
import sys
import subprocess
from typing import Literal
from app.reranker import calc_score
from app.utils import batching
import asyncio
import logging
from bs4 import BeautifulSoup
import re
from playwright.async_api import async_playwright
import resource
import os

logger = logging.getLogger(__name__)

class AdvanceSearchResult(SearchResult):
    def __init__(self, SearchResult: SearchResult, score: float):
        super().__init__(SearchResult.url, SearchResult.title, SearchResult.description)
        self.score = score

    def __repr__(self):
        return f"SearchResult(url={self.url!r}, title={self.title!r}, description={self.description!r}, score={self.score})"

python_toolkit = FastMCP(name="Python-Toolkit")
web_toolkit = FastMCP(name="Web-Toolkit")
bio_toolkit = FastMCP(name="Bio-Toolkit")

def limit_resource(memory_limit: int, cpu_limit: int):
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

@python_toolkit.tool(
    name="run",
    description="Run Python code. Return the result of the code and all declared variables. Use this toolcall for complex tasks like math solving, data analysis, etc.",
    annotations={
        "code": "The Python code to execute",
    }
)
async def python_interpreter(code: str) -> str:
    variables = []
    with open("code.txt", "a") as f:
        f.write(code + '\n')

    tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            first_target = node.targets[0]
            if isinstance(first_target, ast.Name):
                variables.append(first_target.id)
                
            if isinstance(first_target, ast.Attribute):
                variables.append(first_target.attr)
            
            if isinstance(first_target, ast.Subscript):
                variables.append(first_target.value.id)

            if isinstance(first_target, ast.Tuple):
                for target in first_target.elts:
                    if isinstance(target, ast.Name):
                        variables.append(target.id)
                        
            if isinstance(first_target, ast.List):
                for target in first_target.elts:
                    if isinstance(target, ast.Name):
                        variables.append(target.id)

    for var in variables:
        code += f'\nprint("{var} = ", {var})'

    current_interpreter = sys.executable

    result = await asyncio.to_thread(
        subprocess.check_output, 
        [current_interpreter, "-c", code],
        preexec_fn=lambda: limit_resource(100 * 1024 * 1024, 10),
        timeout=30
    )

    return result.decode("utf-8")

@web_toolkit.tool(
    name="search",
    description="Search the web for a given query. Return real-time related information.",
    annotations={
        "query": "The query to search the web for",
        "lang": "The language code of the query",
    }
)
async def search_web(query: str, lang: str = "en") -> list[AdvanceSearchResult | SearchResult]:
    results = list(search(
        query, 
        sleep_interval=5, 
        advanced=True, 
        lang=lang, 
        num_results=10,
    ))

    scores = await asyncio.to_thread(
        calc_score,
        query, 
        [result.description for result in results],
        f'Given a list of senctences, retrieve relevant results that match the query'
    )

    results: list[AdvanceSearchResult] = [
        AdvanceSearchResult(result, score) 
        for result, (_, score) in zip(results, scores)
    ]

    results = sorted(results, key=lambda x: x.score, reverse=True)
    return results[:10]

@web_toolkit.tool(
    name="scrape",
    description="Scrape a URL. Return the content of the page.",
    annotations={
        "url": "The URL to scrape",
    }
)
async def scrape(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor'
            ]
        )
        
        # Create context with realistic user agent and settings
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        
        page = await context.new_page()
        
        # Additional stealth measures
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)
        
        # Navigate and wait for network to be idle
        await page.goto(url, timeout=60000, wait_until="networkidle")
        await page.wait_for_timeout(1000)
        
        content = await page.locator("body").inner_html()
        await browser.close()

        html = BeautifulSoup(content, 'html.parser')
        
        # remove style, script
        for style in html.find_all('style'):
            style.decompose()

        for script in html.find_all('script'):
            script.decompose()

        text = html.get_text(separator=" ")
        
        # remove duplicate spaces
        text = re.sub(r'\s+', ' ', text)

        # remove leading and trailing spaces
        text = text.strip()

        return text

import os
import json

def load_bio() -> dict:
    if not os.path.exists("bio.json"):
        with open("bio.json", "w") as f:
            json.dump({'content': []}, f)

    with open("bio.json", "r") as f:
        return json.load(f)
    
def save_bio(bio_data: dict) -> None:
    with open("bio.json", "w") as f:
        json.dump(bio_data, f)

@bio_toolkit.tool(
    name="action",
    description="Use to manage the user information. Use this tool to manage important information that you want to remember.",
    annotations={
        "action": "The action to perform",
        "content": "The content to be used in the action",
    }
)
async def bio(action: Literal["write", "delete"], content: str) -> bool:
    bio_data = await asyncio.to_thread(load_bio)
    scores = []
    success = False

    for batch in batching(bio_data['content'], 10):
        scores.extend(
            await asyncio.to_thread(
                calc_score,
                content, 
                batch, 
                f'Given a list of sentences, retrieve relevant passages that match the query'
            )
        )

    if action == "write":
        if all(score < 0.5 for _, score in scores):
            bio_data['content'].append(content)
            success = True
    
    elif action == "delete":
        before = len(bio_data['content'])
        bio_data['content'] = [
            item for i,item in enumerate(bio_data['content']) 
            if scores[i][-1] > 0.5
        ]
        after = len(bio_data['content'])
        success = before != after

    await asyncio.to_thread(save_bio, bio_data)
    return success

async def get_bio(query: str) -> list[str]:
    bio_data = load_bio()
    scores = []

    for batch in batching(bio_data['content'], 10):
        scores.extend(
            await asyncio.to_thread(
                calc_score,
                query, 
                batch, 
                f'Given a list of sentences, retrieve relevant results that related to the current user\'s message, in terms of topics, keywords, events, etc.'
            )
        )

    logger.info(f"Bio scores: {scores}")

    return [
        item
        for item, score in scores
        if score > 0.2
    ]

compose = FastMCP(name="Compose")
compose.mount("python", python_toolkit)
compose.mount("web", web_toolkit)
compose.mount("bio", bio_toolkit)