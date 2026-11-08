import asyncio
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, JobContext
from livekit.plugins import groq, silero, openai

load_dotenv()

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=openai.STT(model="whisper-1"),
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=openai.TTS(voice="onyx"),
        vad=silero.VAD.load()
    )

    agent = Agent(
        instructions=(
            "You are JARVIS, a highly intelligent, polite, and witty AI assistant. "
            "Keep your responses concise, natural, and direct for real-time voice conversations."
        )
    )

    await session.start(room=ctx.room, agent=agent)

    await session.generate_reply(
        instructions="Greet the user as JARVIS and ask how you can assist them today."
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
