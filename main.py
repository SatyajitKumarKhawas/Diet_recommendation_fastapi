import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware

# ---------- LIFESTYLE & DIET OPTIMIZER CLASS ----------
class LifestyleDietOptimizer:
    def __init__(self, api_key):
        """Initialize the optimizer with Groq API key"""
        os.environ["GROQ_API_KEY"] = api_key
        self.llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=0.7
        )

    def create_diet_chart_prompt(self):
        return PromptTemplate(
            input_variables=["age", "gender", "activity_level", "health_goals", "dietary_restrictions"],
            template="""
            You are a certified nutritionist and lifestyle coach. 
            Based on the following user details:
            - Age: {age}
            - Gender: {gender}
            - Activity level: {activity_level}
            - Health goals: {health_goals}
            - Dietary restrictions: {dietary_restrictions}

            Create a detailed 7-day diet plan with:
            - Breakfast, lunch, dinner, and 2 snacks per day
            - Calorie estimates
            - Nutritional balance explanation
            """
        )

    def create_lifestyle_prompt(self):
        return PromptTemplate(
            input_variables=["age", "activity_level", "health_goals", "stress_level", "sleep_hours"],
            template="""
            You are a certified lifestyle and wellness coach. 
            Based on:
            - Age: {age}
            - Activity level: {activity_level}
            - Health goals: {health_goals}
            - Stress level: {stress_level}/10
            - Average sleep: {sleep_hours} hours

            Give a weekly lifestyle improvement plan including:
            - Exercise schedule
            - Sleep improvement tips
            - Stress management activities
            - Habit building recommendations
            """
        )

    def generate_diet_plan(self, user_data):
        chain = self.create_diet_chart_prompt() | self.llm
        return chain.invoke(user_data).content

    def generate_lifestyle_plan(self, user_data):
        chain = self.create_lifestyle_prompt() | self.llm
        return chain.invoke(user_data).content

    def generate_comprehensive_plan(self, user_data):
        diet_plan = self.generate_diet_plan(user_data)
        lifestyle_plan = self.generate_lifestyle_plan(user_data)
        return {
            "diet_plan": diet_plan,
            "lifestyle_plan": lifestyle_plan,
            "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# ---------- FASTAPI SETUP ----------
app = FastAPI()

# Allow frontend to connect (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- REQUEST MODEL ----------
class UserInput(BaseModel):
    api_key: str
    age: str
    gender: str
    activity_level: str
    health_goals: str
    dietary_restrictions: str
    stress_level: str
    sleep_hours: str

# ---------- ROUTES ----------
@app.post("/generate-plan")
def generate_plan(data: UserInput):
    try:
        optimizer = LifestyleDietOptimizer(data.api_key)
        plan = optimizer.generate_comprehensive_plan(data.dict())
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Diet & Lifestyle Optimizer API is running!"}
