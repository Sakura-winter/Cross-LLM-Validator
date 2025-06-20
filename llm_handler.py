import asyncio
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import openai
import google.generativeai as genai

# Load environment variables
load_dotenv()

class LLMHandler:
    def __init__(self):
        # Configure both OpenAI and Gemini clients
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file.")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")

        self.openai_client = openai.AsyncOpenAI(api_key=self.openai_api_key)
        genai.configure(api_key=self.gemini_api_key)
        
        self.models = [
            {"name": "GPT-4o-mini", "model_id": "gpt-4o-mini", "api": "openai"},
            {"name": "Gemini-1.5-Flash", "model_id": "gemini-1.5-flash-latest", "api": "gemini"}
        ]
        self.judge_model_id = "gpt-4o-mini"
        self.judge_api = "openai"

    async def _call_openai(self, model_id: str, messages: List[Dict[str, str]], temp: float) -> Dict:
        try:
            response = await self.openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temp
            )
            return {"success": True, "content": response.choices[0].message.content}
        except Exception as e:
            return {"success": False, "error": f"OpenAI Error: {e}"}

    def _call_gemini_sync(self, model_id: str, messages: List[Dict[str, str]], temp: float) -> Dict:
        """This is now a synchronous function to be run in a separate thread."""
        try:
            model = genai.GenerativeModel(model_id)
            # Gemini's sync API has a different prompt format requirement
            gemini_messages = [m['content'] for m in messages if m['role'] == 'user']
            prompt = "\\n".join(gemini_messages)
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temp)
            )
            return {"success": True, "content": response.text}
        except Exception as e:
            return {"success": False, "error": f"Gemini Error: {e}"}

    async def _make_api_call(self, api: str, model_id: str, messages: List[Dict[str, str]], temp: float = 0.7) -> Dict:
        if api == "openai":
            return await self._call_openai(model_id, messages, temp)
        elif api == "gemini":
            # Run the synchronous Gemini function in a separate thread to avoid event loop conflicts
            return await asyncio.to_thread(self._call_gemini_sync, model_id, messages, temp)
        else:
            return {"success": False, "error": f"Unknown API: {api}"}

    async def get_initial_answers(self, question: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": question}]
        tasks = [self._make_api_call(m["api"], m["model_id"], messages) for m in self.models]
        results = await asyncio.gather(*tasks)
        
        answers = {}
        for i, result in enumerate(results):
            model_info = self.models[i]
            answers[model_info["name"]] = {**result, **model_info}
        return answers

    async def are_answers_similar(self, question: str, answers: Dict[str, Any]) -> bool:
        answer_list = [f"Answer from {data['name']}:\\n{data['content']}" for model, data in answers.items() if 'content' in data]
        if len(answer_list) < 2:
            return True

        comparison_prompt = f"""The user asked: "{question}"
I received two answers:
---
{answer_list[0]}
---
{answer_list[1]}
---
Do these answers provide the same core information? Respond with only YES or NO."""
        
        messages = [{"role": "user", "content": comparison_prompt}]
        result = await self._make_api_call(self.judge_api, self.judge_model_id, messages, temp=0)
        return result.get("content", "").strip().upper() == "YES"

    async def get_revised_answers(self, question: str, original_answers: Dict[str, Any]) -> Dict[str, Any]:
        tasks = []
        for model_to_revise in self.models:
            model_name = model_to_revise["name"]
            other_model = next((m for m in self.models if m["name"] != model_name), None)
            
            if not other_model or model_name not in original_answers or other_model["name"] not in original_answers:
                continue

            feedback_prompt = f"""Original question: "{question}"
Your answer: "{original_answers[model_name]['content']}"
The other model's answer: "{original_answers[other_model['name']]['content']}"
Review both. If yours is best, defend it. Otherwise, provide a revised answer. Respond with only the final answer."""

            messages = [{"role": "user", "content": feedback_prompt}]
            task = self._make_api_call(model_to_revise["api"], model_to_revise["model_id"], messages)
            tasks.append((model_name, task))

        results = await asyncio.gather(*[task for _, task in tasks])
        
        revised_answers = {}
        for i, (model_name, _) in enumerate(tasks):
            result = results[i]
            if result["success"]:
                revised_answers[model_name] = {**original_answers[model_name], "content": result["content"], "original_content": original_answers[model_name].get("content")}
            else:
                revised_answers[model_name] = {**original_answers[model_name], "error": result["error"]}
        
        return revised_answers
    
    def get_models_list(self) -> List[Dict[str, str]]:
        return self.models.copy()
    
    def add_model(self, name: str, model_id: str, display_name: str = None):
        """Add a new model to the configuration"""
        if display_name is None:
            display_name = name
        
        self.models.append({
            "name": name,
            "model_id": model_id,
            "display_name": display_name
        })
    
    def remove_model(self, name: str):
        """Remove a model from the configuration"""
        self.models = [model for model in self.models if model["name"] != name] 