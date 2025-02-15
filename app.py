from typing import List  

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.mistral import MistralModel
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from pydantic_ai.models.groq import GroqModel



load_dotenv()

topic = "Deepseek R1"

class BlogOutput(BaseModel):
    Heading : str
    Content : str

class FeedbackAnalysis(BaseModel):
    approval_status: bool  # 'Approved' or 'Not Approved'
    change_required: bool  # 'Yes' or 'No'
    requested_changes: List[str] # List of change requests


#model = MistralModel('mistral-small-latest', api_key=os.getenv('MISTRAL_API_KEY'))
model1 = GroqModel('llama3-70b-8192', api_key=os.getenv('GROQ_API_KEY'))

agent = Agent(model1, deps_type= str, 
              result_type= BlogOutput,
              system_prompt='You are a quirky Blog creator who simplifies stuff and writes blogs about the given topic. You sound extremely human and nerdy and excited')

@agent.system_prompt
def add_topic(ctx:RunContext[str]) -> str:
    return f"The topic is {ctx.deps}"

@agent.tool # DDGS
async def get_search(search_data:RunContext[str],query: str) -> dict[str, str]:
    """Get the search for a keyword query.

    Args:
        query: keywords to search.
    """
    print(f"Search query: {query}")
    max_results = 3
    results =  DDGS(proxy=None).text(query, max_results=max_results)
    # print("RESB",results)

    return results

result = agent.run_sync("talk about", deps = topic)

# print(result.data)
print(result.data)

print("Hey! sent you the blog post for today! What do you think?")
opinion = input()


model2 = GroqModel('llama3-70b-8192', api_key=os.getenv('GROQ_API_KEY'))

agent2 = Agent(model2, deps_type= str, 
              result_type= FeedbackAnalysis,
              system_prompt=(
        "You are a structured feedback analyzer. Your job is to analyze user feedback on blog content."
        " You must follow these rules:\n"
        " 1. If the user approves the content, set 'approval_status' to True. Otherwise, set it to False.\n"
        " 2. If the user requests ANY modifications (explicitly or implicitly), set 'change_required' to True. Otherwise, set it to False.\n"
        " 3. If 'change_required' is True, you must populate 'requested_changes' with a list of exact requested modifications.\n"
        
        " 5. DO NOT leave 'requested_changes' empty if 'change_required' is True.\n"
))
@agent2.system_prompt
def add_topic(ctx:RunContext[str]) -> str:
    return f"The opinion is {ctx.deps}"

opinionResult = agent2.run_sync("Analyze the feedback:",deps = opinion)
print(opinionResult.data)

print(result.all_messages())
if opinionResult.data.change_required == True:
    @agent.system_prompt
    def add_feedback(ctx:RunContext[str]) ->str:
        feedback_prompt = "The user's feedback for the blog is "
        for ever in opinionResult.data.requested_changes:
            feedback_prompt += ever
        return feedback_prompt


updatedBlog = agent.run_sync("Give updated blog on the given topic, based on the feedback", message_history=result.all_messages())


# print(updatedBlog.data)



