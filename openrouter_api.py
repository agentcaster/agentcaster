import os
import requests
import json
import base64

class OpenRouterAPI:
    def __init__(self, api_key_file='openrouter_api_key.txt'):
        with open(api_key_file, 'r') as f:
            self.api_key = f.read().strip()
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "openai/gpt-4o-mini"
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def chat_completion(self, messages, tools=None, tool_choice=None, stream=False):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://agentcaster.ai",
            "X-Title": "AgentCaster"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream
        }
        
        if tools:
            payload["tools"] = tools
        
        if tool_choice:
            payload["tool_choice"] = tool_choice
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        return response.json()
    
    def add_image_to_message(self, message, image_path):
        image_base64 = self.encode_image(image_path)
        
        if isinstance(message["content"], list):
            message["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            })
        else:
            text_content = message["content"]
            message["content"] = [
                {
                    "type": "text",
                    "text": text_content
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        
        return message
