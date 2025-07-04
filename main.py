import os
import asyncio
import chainlit as cl
from dotenv import load_dotenv
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, Agent, Runner,RunConfig


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

chat_model = OpenAIChatCompletionsModel(
    openai_client=client,
    model="gemini-2.0-flash"
)

config = RunConfig(
    model=chat_model,
    model_provider=client,
    tracing_disabled=True
)

web_dev_agent = Agent(
    name="Web Developer Agent",
    instructions="Handle queries about web development: HTML, CSS, JS, React, Next.js, Python, Tailwind, Bootstrap etc."
)

app_dev_agent = Agent(
    name="App Developer Agent",
    instructions="Handle mobile app development: Flutter, React Native etc."
)

marketing_agent = Agent(
    name="Marketing Agent",
    instructions="Assist with digital marketing: SEO, ads, social media, content strategies etc."
)

manager = Agent(
    name="Manager Agent",
    instructions="Figure out which category the user's query belongs to and route it to the relevant agent.",
    handoffs=[web_dev_agent, app_dev_agent, marketing_agent]
)

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="👋 Hello, I'm the Gemini Manager Agent. How can I help you today?\nAsk me anything about:\n- 🌐 Web Development\n- 📱 App Development\n- 📈 Digital Marketing"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")

    # Add current msg to history
    history.append({"role": "user", "content": message.content})
    cl.user_session.set("history", history)

    msg = cl.Message(content="⠋ Typing...")
    await msg.send()

    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    spinner_running = True

    async def spinner_task():
        idx = 0
        while spinner_running:
            frame = spinner_frames[idx % len(spinner_frames)]
            msg.content = f"{frame} Thinking..."
            await msg.update()
            idx += 1
            await asyncio.sleep(0.1)

    # start spinner task
    spinner = asyncio.create_task(spinner_task())

    # Run the manager agent
    result = await Runner.run(
        manager,
        input=history,
        run_config=config
    )

    # stop spinner
    spinner_running = False
    await spinner  # wait for it to finish

  
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

  
    final_content = ""
    idx = 0
    for line in result.final_output.splitlines():
        for char in line + "\n":
            final_content += char
            frame = spinner_frames[idx % len(spinner_frames)]  # rotate spinner

            msg.content = f"{frame} {final_content}"
            await msg.update()

            idx += 1
            await asyncio.sleep(0.01)

    # Final update: remove spinner
    msg.content = final_content
    await msg.update()
