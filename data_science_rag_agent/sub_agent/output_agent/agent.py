from google.adk.agents import Agent
from pydantic import BaseModel, Field
from typing import List


class Section(BaseModel):
    heading: str = Field(
        description=(
            "Main title of this section. "
            "It represents the core concept or topic being explained "
            "based on the user's query."
        )
    )
    sub_heading: str = Field(
        description=(
            "A smaller contextual title under the main heading. "
            "It provides extra clarity or categorization for the section's content."
        )
    )
    content: str = Field(
        description=(
            "A detailed explanation or theory of the topic in this section. "
            "This should include clear, simple explanations tailored to the user's skill level "
            "(beginner, intermediate, or professional), and can cover concepts, reasoning, "
            "or explanations of code provided in this section."
        )
    )
    code_blocks: List[str] = Field(
        description=(
            "A list of clean and executable code snippets provided as part of this section. "
            "Each snippet should be directly relevant to the user's query "
            "and easy to understand, following best practices."
        )
    )
    examples: List[str] = Field(
        description=(
            "A practical example or demonstration of how the concept or code is applied. "
            "This helps users connect the explanation with real-world use cases."
        )
    )
    key_points: List[str] = Field(
        description=(
            "Concise bullet points highlighting key takeaways, "
            "important concepts, or quick tips from this section."
        )
    )
    notes: str = Field(
        description=(
            "Optional additional remarks, pro tips, or warnings "
            "to guide the user toward better understanding or to prevent common mistakes."
        )
    )


class AgentResponseSchema(BaseModel):
    title: str = Field(
        description=(
            "The main title or overall topic of the agent's response. "
            "This should summarize the purpose of the answer clearly "
            "and be easy to display in the frontend."
        )
    )
    sections: List[Section] = Field(
        description=(
            "A list of structured sections that divide the response "
            "into logical, easy-to-read blocks. "
            "Each section may include headings, explanations, code, examples, "
            "key points, and optional notes for a comprehensive answer."
        )
    )


output_agent = Agent(
    name="output_agent",
    description="Convert the Agent Response into valid well structured json",
    instruction="""
    You are a helpful agent which whose task is to accept the reponse from the root_agent and convert it into the valid json data
    You have to follow the schema
    Example:

    your output: 
            {
  "title": "Sample Agent Response",
  "sections": [
    {
      "heading": "Main Concept or Topic",
      "sub_heading": "Contextual Sub-Heading",
      "content": "This is a detailed explanation of the topic in this section. It is tailored to the user's skill level (beginner, intermediate, or professional). It may include reasoning, conceptual details, or explanations of provided code examples.",
      "code_blocks": [
        "print('Hello, World!')",
        "import pandas as pd\ndata = pd.Series([1, 2, 3])\nprint(data.mean())"
      ],
      "examples": [
        "Explaining logistic regression with a dataset of customer churn.",
        "Using pandas to calculate basic statistics like mean or median."
      ],
      "key_points": [
        "Always tailor responses to the user’s level.",
        "Use clear and runnable code examples.",
        "Provide real-world scenarios for better understanding."
      ],
      "notes": "Ensure clarity and accuracy by grounding responses in reliable data."
    }
  ]
}


""",
    output_schema=AgentResponseSchema,
    output_key="AgentResponse",
)
