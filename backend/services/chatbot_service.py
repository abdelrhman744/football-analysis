"""
Chatbot Analysis Service
Provides AI-powered match analysis via a conversational interface.

TODO: Replace placeholder responses with a real LLM integration.
      Suggested approaches:
      - Anthropic Claude API (claude-sonnet-4-20250514) with full match context
      - OpenAI GPT-4 with function calling for structured data retrieval
      - Local LLM (Llama 3, Mistral) via Ollama for self-hosted solution

Example Claude API integration:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def build_system_prompt(analysis: FullAnalysisResult) -> str:
        return f\"\"\"You are an expert football analyst.
        Match data: {analysis.model_dump_json()}
        Provide detailed tactical analysis based on this data.
        \"\"\"

    def ask_claude(question: str, analysis: FullAnalysisResult, history: list) -> str:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=build_system_prompt(analysis),
            messages=[*history, {"role": "user", "content": question}]
        )
        return response.content[0].text
"""

import random
from models.schemas import ChatRequest, ChatResponse


# Placeholder responses keyed by common football analysis topics
PLACEHOLDER_RESPONSES = {
    "possession": (
        "Based on the match analysis, Team A dominated possession with approximately "
        "58% of ball time. Their build-up play was concentrated through the center, "
        "with the defensive midfielder acting as the primary ball distributor."
    ),
    "pressing": (
        "Team B showed an aggressive high-press in the first half, forcing several "
        "turnovers in Team A's defensive third. However, the press intensity dropped "
        "significantly after the 60-minute mark, allowing Team A more time on the ball."
    ),
    "heatmap": (
        "The heatmap data reveals Team A's wide attackers had high activity in the "
        "final third channels, while Team B's wingers sat deeper, suggesting a "
        "defensively oriented 4-4-2 mid-block structure."
    ),
    "defense": (
        "Defensively, both teams maintained compact shape. Team A's defensive line "
        "averaged a relatively high position, leaving space behind for counter-attacks. "
        "Team B exploited this space twice, generating their best chances of the match."
    ),
    "attack": (
        "Team A's attacking patterns show a preference for combination play through "
        "the half-spaces. The number 10 operated between the lines effectively, "
        "serving as the link between midfield and the striker."
    ),
    "default": (
        "Based on the comprehensive match analysis including player tracking data, "
        "movement heatmaps, and team classification results, this was a tactically "
        "balanced match. Team A showed stronger possession-based control, while "
        "Team B's direct approach created dangerous transitional moments. "
        "I recommend focusing on the heatmap data for deeper spatial insights."
    ),
}


def get_placeholder_response(question: str) -> str:
    """
    Return a contextually relevant placeholder answer.

    TODO: Replace this entire function with real LLM API call.
    The real implementation should:
    1. Load the full FullAnalysisResult for the given video_id
    2. Build a detailed system prompt with all match data
    3. Send the user's question + conversation history to the LLM
    4. Return the LLM's structured response
    """
    question_lower = question.lower()
    for keyword, response in PLACEHOLDER_RESPONSES.items():
        if keyword in question_lower:
            return response
    return PLACEHOLDER_RESPONSES["default"]


async def generate_chat_response(request: ChatRequest) -> ChatResponse:
    """
    Generate an AI chatbot response based on the user's question and match data.

    TODO: Replace placeholder logic with real LLM integration.

    Steps for real implementation:
    1. Load the FullAnalysisResult for request.video_id from results storage
    2. Serialize all match data (detection, tracking, heatmap, stats, summary)
    3. Build system prompt with full match context
    4. Call LLM API with conversation history + new question
    5. Parse and return the structured response

    Args:
        request: ChatRequest with video_id, question, and conversation history

    Returns:
        ChatResponse with the AI-generated answer
    """

    # ── PLACEHOLDER: Return context-aware dummy response ──────────────────────
    answer = get_placeholder_response(request.question)
    sources_used = random.sample(
        ["detection_data", "tracking_data", "heatmap_data", "match_stats"],
        k=random.randint(1, 3),
    )
    # ── END PLACEHOLDER ───────────────────────────────────────────────────────

    return ChatResponse(
        answer=answer,
        video_id=request.video_id,
        sources_used=sources_used,
    )
