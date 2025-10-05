"""
COMPLETE MEDICAL HISTORY SYSTEM - INTEGRATION GUIDE
Shows how LangGraph structure + Urdu configuration work together
"""

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
import json

# ========== FILE 1: STATE & STRUCTURE (LangGraph) ==========
# This defines HOW the conversation flows

class HistoryState(TypedDict):
    """The state that flows through the graph"""
    messages: List[dict]
    current_section: str
    collected_data: dict
    section_complete: bool
    all_sections_done: bool
    language_preference: str  # Added for language handling


# ========== FILE 2: LANGUAGE & PROMPTS (Urdu Config) ==========
# This defines WHAT the agent says and HOW it communicates

class UrduPromptBuilder:
    """Builds Urdu prompts for each section"""
    
    URDU_BASE_PROMPT = """You are a medical history-taking assistant conducting interviews in URDU (اردو).

CRITICAL LANGUAGE RULES:
1. **Always respond in Urdu script (اردو رسم الخط)** unless patient uses Roman Urdu
2. **Use simple, conversational Urdu** - avoid complex vocabulary
3. **Mix common English medical terms** naturally (blood pressure, diabetes, X-ray)
4. **Use respectful آپ (aap) form**
5. **Accept mixed language** - mirror patient's style
6. **Dont make a diagnosis or give suggestions, Only ask about the patient history**

CURRENT SECTION: {current_section}
COLLECTED DATA SO FAR: {collected_data}

{section_specific_prompt}

Continue the interview naturally in Urdu."""

    SECTION_PROMPTS = {
        "patient_name": """
**CURRENT TASK: Get patient's full name**

INSTRUCTIONS:
- If data NOT recorded and FIRST interaction: Ask "آپ کا پورا نام کیا ہے؟" (What is your full name?)
- If data NOT recorded and user gave UNCLEAR response: Ask again "معاف کریں، براہ کرم اپنا صحیح نام بتائیں؟"
- If data NOT recorded and user gave CLEAR name: Use RecordInfo tool, then MarkSectionComplete
- If data ALREADY recorded: Use MarkSectionComplete to move forward

VALIDATION RULES:
- CLEAR: "احمد علی", "فاطمہ خان", "Muhammad Ali" 
- UNCLEAR: "موسیقی", "abc", "123", single letters, gibberish

**IMPORTANT: Only ask for name if not already recorded.**
""",
        "patient_age": """
**CURRENT TASK: Get patient's age**

INSTRUCTIONS:
- Ask "آپ کی عمر کتنی ہے؟" (What is your age?)
- If unclear: "براہ کرم اپنی عمر سالوں میں بتائیں؟"
- If clear: Record and mark complete

VALIDATION:
- CLEAR: "25", "پچیس سال", "25 years", "تیس"
- UNCLEAR: random words, non-age responses

**Only ask for age. One question at a time.**
""",
        "patient_gender": """
**CURRENT TASK: Get patient's gender**

INSTRUCTIONS:
- Ask "آپ مرد ہیں یا خاتون؟" (Are you male or female?)
- If unclear: Ask again politely
- If clear: Record and mark complete

VALIDATION:
- CLEAR: "مرد", "خاتون", "male", "female", "man", "woman"
- UNCLEAR: unrelated responses

**Only ask for gender.**
""",
        "patient_occupation": """
**CURRENT TASK: Get patient's occupation**

INSTRUCTIONS:
- Ask "آپ کیا کام کرتے ہیں؟" (What work do you do?)
- Accept: jobs, "student", "گھریلو خاتون", "ریٹائرڈ"
- If unclear: Ask again

**Only ask for occupation.**
""",
        "patient_address": """
**CURRENT TASK: Get patient's address**

INSTRUCTIONS:
- Ask "آپ کہاں رہتے ہیں؟ شہر یا علاقہ بتائیں؟"
- Accept: city names, areas, districts
- If unclear: Ask again

**Only ask for address/location.**
""",
        "patient_contact": """
**CURRENT TASK: Get patient's contact number**

INSTRUCTIONS:
- Ask "آپ کا فون نمبر کیا ہے؟"
- Accept: phone numbers (any format)
- If unclear: Ask again

**Only ask for contact number.**
""",
        "complaint": """
**CURRENT TASK: Get chief complaint**

INSTRUCTIONS:
- If data NOT recorded: Ask "آپ کو کیا تکلیف ہے؟ کیا مسئلہ ہے؟"
- If data ALREADY recorded: Use MarkSectionComplete to move forward
- Listen for main health problem
- If unclear: Ask them to describe their main problem

**Only ask for main complaint if not already recorded.**
""",
        "hpc_pain": """
Based on complaint, ask relevant SOCRATES questions in Urdu:
- کہاں درد ہے؟ (Site)
- کب شروع ہوا؟ (Onset)  
- کیسا درد ہے؟ (Character)
""",
        "systems": """
Relevant system review in Urdu based on complaint
""",
        "pmh": """
پرانی بیماریاں: Sugar? Pressure? دل کی بیماری؟
""",
        "drugs": """
کوئی دوائیں؟ Allergies?
""",
        "social": """
سگریٹ؟ رہائش? مدد کی ضرورت؟
"""
    }
    
    @staticmethod
    def build_prompt(section: str, collected_data: dict, last_user_response: str = None) -> str:
        """Build section-specific Urdu prompt with validation"""
        section_prompt = UrduPromptBuilder.SECTION_PROMPTS.get(section, "")
        
        base_prompt = f"""You are a medical history-taking assistant conducting interviews in URDU (اردو).

**LANGUAGE RULES:**
1. Always respond in Urdu script (اردو رسم الخط)
2. Use simple, conversational Urdu
3. Use respectful آپ (aap) form
4. Mix common English medical terms naturally

**CONVERSATION RULES:**
1. **ASK ONE QUESTION AT A TIME** - Focus only on current section
2. **VALIDATE RESPONSES** - Check if answer makes sense for the question
3. **BE PATIENT** - If unclear, ask same question again politely
4. **USE TOOLS ONLY ONCE** - When you get a clear answer:
   - First use RecordInfo tool to save the data
   - Then use MarkSectionComplete tool to move forward
   - DO NOT call tools multiple times for same data

**CURRENT SECTION:** {section}
**COLLECTED DATA SO FAR:**
{json.dumps(collected_data, ensure_ascii=False, indent=2)}

{section_prompt}"""

        # Add validation context if there's a previous user response
        if last_user_response:
            # Check if current section data is already recorded
            section_mapping = {
                'patient_name': ('demographics', 'name'),
                'patient_age': ('demographics', 'age'),
                'patient_gender': ('demographics', 'gender'),
                'patient_occupation': ('demographics', 'occupation'),
                'patient_address': ('demographics', 'address'),
                'patient_contact': ('demographics', 'contact'),
                'complaint': ('presentation', 'chief_complaint'),
                'hpc_pain': ('history', 'hpc'),
                'systems': ('review', 'systems'),
                'pmh': ('history', 'past_medical'),
                'drugs': ('medications', 'current'),
                'social': ('social', 'history')
            }
            
            data_already_recorded = False
            if section in section_mapping:
                mapped_section, mapped_field = section_mapping[section]
                if (mapped_section in collected_data and 
                    mapped_field in collected_data[mapped_section] and
                    collected_data[mapped_section][mapped_field]):  # Check not empty
                    data_already_recorded = True
            
            base_prompt += f"""

**VALIDATION CONTEXT:**
Last user response: "{last_user_response}"
Data already recorded for {section}: {data_already_recorded}

**CRITICAL DECISION LOGIC:**
1. If data is ALREADY RECORDED for {section}:
   - ONLY use MarkSectionComplete to move forward
   - DO NOT use RecordInfo again
   - DO NOT ask the same question again
2. If "{last_user_response}" is clear and relevant for {section} AND data not recorded:
   - Use RecordInfo tool ONCE to save the data
   - Then use MarkSectionComplete tool ONCE to move forward
3. If UNCLEAR or IRRELEVANT and data not recorded:
   - Ask the same question again politely
   - DO NOT use any tools

**REMEMBER: If data is already recorded, just mark complete and move on!**
- Examples of UNCLEAR: "موسیقی", random words, gibberish, unrelated answers

**Remember: One clear question, wait for clear answer, validate, then proceed.**"""
        
        return base_prompt
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect patient's language preference"""
        # Urdu script check
        if any('\u0600' <= c <= '\u06FF' for c in text):
            return 'urdu_script'
        
        # Roman Urdu check
        roman_urdu_words = ['hai', 'mein', 'ka', 'dard', 'bukhar']
        if any(word in text.lower() for word in roman_urdu_words):
            return 'roman_urdu'
        
        return 'english'


# ========== TOOLS: Structure the data ==========

class RecordInfo(BaseModel):
    """Generic tool to record information"""
    section: str = Field(description="demographics/complaint/hpc/systems/pmh/drugs/social")
    field: str = Field(description="Specific field name")
    value: str = Field(description="The value to record")

class MarkSectionComplete(BaseModel):
    """Mark section as complete"""
    section: str
    reasoning: str


# ========== INTEGRATION: Bringing it all together ==========

class UrduMedicalHistorySystem:
    """
    THIS IS THE MAIN CLASS THAT COMBINES EVERYTHING
    - Uses LangGraph for flow control (File 1)
    - Uses Urdu prompts for communication (File 2)
    """
    
    def __init__(self):
        # Initialize LLM using official langchain-groq ChatGroq
        api_key = os.getenv('GROQ_API_KEY')
        
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # Use official ChatGroq with Urdu-optimized settings
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",  # Good for multilingual/Urdu
            temperature=0.3,
            groq_api_key=api_key,
            max_tokens=1024
        )
        
        # Create and bind tools
        tools = [RecordInfo, MarkSectionComplete]
        self.llm_with_tools = self.llm.bind_tools(tools)
        
        # Initialize prompt builder
        self.prompt_builder = UrduPromptBuilder()
        
        # Section order
        self.sections_order = [
            'patient_name',
            'patient_age',
            'patient_gender',
            'patient_occupation',
            'patient_address',
            'patient_contact',
            'complaint', 
            'hpc_pain',
            'systems',
            'pmh',
            'drugs',
            'social'
        ]
        
        # Build the graph
        self.graph = self._build_graph()
    
    
    # ========== GRAPH NODES ==========
    # These are from File 1 (LangGraph structure)
    
    def agent_node(self, state: HistoryState) -> HistoryState:
        """
        CORE NODE: Where LLM makes decisions with response validation
        This is where File 1 (structure) meets File 2 (language)
        """
        
        # Skip if section is already complete - let router handle transition
        if state['section_complete']:
            print(f"⏭️ Section {state['current_section']} already complete, skipping agent")
            return state
        
        # Get last user response for validation
        last_user_response = None
        user_messages = [m for m in state['messages'] if m['role'] == 'user']
        if user_messages:
            last_user_response = user_messages[-1]['content']
        
        # BUILD URDU PROMPT with validation context
        system_prompt = self.prompt_builder.build_prompt(
            section=state['current_section'],
            collected_data=state['collected_data'],
            last_user_response=last_user_response
        )
        
        # PREPARE MESSAGES
        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history (limit to recent exchanges)
        recent_messages = state['messages'][-6:]  # Last 6 messages to avoid token limit
        for msg in recent_messages:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            else:
                messages.append(AIMessage(content=msg['content']))
        
        # CALL LLM (with Urdu instructions and validation)
        response = self.llm_with_tools.invoke(messages)
        
        # UPDATE STATE
        state['messages'].append({
            'role': 'assistant',
            'content': response.content,
            'tool_calls': getattr(response, 'tool_calls', [])
        })
        
        return state
    
    
    def tool_node(self, state: HistoryState) -> HistoryState:
        """
        Execute tools with proper validation and data structuring
        """
        last_message = state['messages'][-1]
        tool_calls = last_message.get('tool_calls', [])
        
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_input = tool_call['args']
            
            if tool_name == 'RecordInfo':
                # Store structured data with section mapping
                section = tool_input.get('section', 'general')
                field = tool_input.get('field', 'unknown')
                value = tool_input.get('value', '')
                
                # Map current section to appropriate data structure
                section_mapping = {
                    'patient_name': ('demographics', 'name'),
                    'patient_age': ('demographics', 'age'),
                    'patient_gender': ('demographics', 'gender'),
                    'patient_occupation': ('demographics', 'occupation'),
                    'patient_address': ('demographics', 'address'),
                    'patient_contact': ('demographics', 'contact'),
                    'complaint': ('presentation', 'chief_complaint'),
                    'hpc_pain': ('history', 'hpc'),
                    'systems': ('review', 'systems'),
                    'pmh': ('history', 'past_medical'),
                    'drugs': ('medications', 'current'),
                    'social': ('social', 'history')
                }
                
                if state['current_section'] in section_mapping:
                    mapped_section, mapped_field = section_mapping[state['current_section']]
                    section = mapped_section
                    field = mapped_field
                
                # Store the data (prevent duplicates with better logic)
                if section not in state['collected_data']:
                    state['collected_data'][section] = {}
                
                # Only record if not already recorded, empty, or significantly different
                current_value = state['collected_data'][section].get(field, '')
                if not current_value or current_value != value:
                    state['collected_data'][section][field] = value
                    print(f"✅ Recorded: {section}.{field} = '{value}'")
                    
                    # Add system message to prevent re-recording
                    state['messages'].append({
                        'role': 'system',
                        'content': f"Data recorded for {state['current_section']}. Ready to move to next section.",
                        'tool_calls': []
                    })
                else:
                    print(f"⚠️ Skipped duplicate: {section}.{field} already contains '{current_value}'")
                    # Don't add system message for duplicates to avoid confusion
            
            elif tool_name == 'MarkSectionComplete':
                if not state['section_complete']:  # Only if not already complete
                    state['section_complete'] = True
                    print(f"✓ Section complete: {state['current_section']}")
                    # Add a message to prevent further tool calls
                    state['messages'].append({
                        'role': 'system',
                        'content': f"Section {state['current_section']} marked complete. Moving to next section.",
                        'tool_calls': []
                    })
                else:
                    print(f"⚠️ Section {state['current_section']} already marked complete - ignoring duplicate call")
        
        return state
    
    
    def next_section_node(self, state: HistoryState) -> HistoryState:
        """Move to next section"""
        current_idx = self.sections_order.index(state['current_section'])
        
        if current_idx < len(self.sections_order) - 1:
            old_section = state['current_section']
            state['current_section'] = self.sections_order[current_idx + 1]
            state['section_complete'] = False  # Reset for next section
            print(f"📋 Moved from '{old_section}' to '{state['current_section']}'")
        else:
            state['all_sections_done'] = True
            print("✅ All sections completed!")
        
        return state
    
    
    def router(self, state: HistoryState) -> str:
        """Decide what to do next"""
        last_msg = state['messages'][-1] if state['messages'] else None
        
        # Check if all sections are done first
        if state['all_sections_done']:
            return "end"
        
        # If section is complete, move to next section
        if state['section_complete']:
            print(f"🔄 Moving from {state['current_section']} to next section")
            return "next_section"
        
        # If LLM called tools, execute them
        if last_msg and last_msg.get('tool_calls'):
            return "tools"
        
        # Otherwise, continue conversation (wait for user or ask question)
        return "continue"
    
    
    # ========== BUILD GRAPH ==========
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(HistoryState)
        
        # Add nodes
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("next_section", self.next_section_node)
        
        # Entry point
        workflow.set_entry_point("agent")
        
        # Routing
        workflow.add_conditional_edges(
            "agent",
            self.router,
            {
                "tools": "tools",
                "next_section": "next_section",
                "continue": END,
                "end": END
            }
        )
        
        workflow.add_edge("tools", "agent")
        workflow.add_edge("next_section", "agent")
        
        # Compile without config parameter for compatibility
        return workflow.compile()
    
    
    # ========== PUBLIC API ==========
    
    def start_interview(self) -> dict:
        """Initialize a new interview"""
        state = {
            "messages": [],
            "current_section": "patient_name",  # Start with patient name
            "collected_data": {},
            "section_complete": False,
            "all_sections_done": False,
            "language_preference": "urdu_script"
        }
        
        # Get first question
        result = self.graph.invoke(state)
        return {
            "ai_message": result['messages'][-1]['content'],
            "state": result
        }
    
    
    def process_user_message(self, state: dict, user_message: str) -> dict:
        """
        Process a user message
        THIS IS YOUR MAIN INTERFACE
        """
        
        # Detect language on first message
        if len(state['messages']) == 1:
            lang_pref = self.prompt_builder.detect_language(user_message)
            state['language_preference'] = lang_pref
        
        # Add user message to state
        state['messages'].append({
            'role': 'user',
            'content': user_message
        })
        
        # Run through graph
        result = self.graph.invoke(state)
        return {
            "ai_message": result['messages'][-1]['content'],
            "state": result,
            "collected_data": result['collected_data'],
            "is_complete": result['all_sections_done']
        }
    
    
    async def process_user_message_streaming(self, state: dict, user_message: str):
        """
        Streaming version for better UX
        Yields tokens as they arrive
        """
        state['messages'].append({
            'role': 'user',
            'content': user_message
        })
        
        # Stream the response
        async for chunk in self.graph.astream(state):
            if 'agent' in chunk:
                yield chunk['agent']

    # ========== TRANSLATION / VIEW HELPERS ==========
    def translate_to_english(self, text: str) -> str:
        """Translate a piece of text to English using the LLM.

        This is used to present a doctor-facing view where all content
        must be in English. The translator preserves medical terms.
        """
        # Build a small translation system prompt
        system_prompt = (
            "You are a helpful translator. Translate the user's text to English. "
            "Preserve medical and clinical terms (do not paraphrase) and keep the meaning exact. "
            "Output only the translated text."
        )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=text)]

        # Use the same llm binding (tools are available but not required)
        response = self.llm_with_tools.invoke(messages)

        return getattr(response, 'content', str(response))

    def get_history_view(self, state: dict, view: str = 'patient') -> List[dict]:
        """Return the conversation history formatted for a specific view.

        - view='patient' returns messages in the patient's preferred language
        - view='doctor' returns messages translated to English (doctor always sees English)

        Note: translation is performed only when necessary (non-English characters detected).
        """
        formatted: List[dict] = []

        for msg in state.get('messages', []):
            content = msg.get('content', '')

            if view == 'doctor':
                # If the message contains Arabic/Urdu script, translate it
                if any('\u0600' <= c <= '\u06FF' for c in content):
                    try:
                        content_en = self.translate_to_english(content)
                        content = content_en
                    except Exception:
                        # Fallback to original if translation fails
                        pass

            # For patient view we assume the messages are already in the preferred language
            formatted.append({'role': msg.get('role', ''), 'content': content})

        return formatted


# ========== USAGE EXAMPLE: How you'd actually use this ==========

def main():
    """
    THIS IS HOW YOU USE THE COMPLETE SYSTEM
    """
    
    # Initialize the system
    system = UrduMedicalHistorySystem()
    
    print("="*60)
    print("URDU MEDICAL HISTORY TAKING SYSTEM")
    print("="*60)
    
    # Start interview
    result = system.start_interview()
    print(f"\nAI: {result['ai_message']}\n")
    
    state = result['state']
    
    # Conversation loop
    while not state['all_sections_done']:
        # Get user input
        user_input = input("Patient: ")
        print()
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        # Process message
        result = system.process_user_message(state, user_input)
        
        # Update state
        state = result['state']
        
        # Show AI response
        print(f"AI: {result['ai_message']}\n")
        
        # Show collected data (for debugging)
        if result['collected_data']:
            print("--- Collected Data ---")
            print(json.dumps(result['collected_data'], ensure_ascii=False, indent=2))
            print("----------------------\n")
    
    # Final summary
    print("\n" + "="*60)
    print("INTERVIEW COMPLETE")
    print("="*60)
    print("\nFinal Collected Data:")
    print(json.dumps(state['collected_data'], ensure_ascii=False, indent=2))


# ========== FASTAPI INTEGRATION (For Production) ==========

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

app = FastAPI()

# Global system instance
medical_system = UrduMedicalHistorySystem()

# Store active sessions
sessions = {}


@app.post("/api/start-interview")
async def start_interview():
    """Start a new interview"""
    result = medical_system.start_interview()
    
    # Generate session ID
    import uuid
    session_id = str(uuid.uuid4())
    
    # Store session
    sessions[session_id] = result['state']
    
    return {
        "session_id": session_id,
        "message": result['ai_message']
    }


@app.post("/api/send-message")
async def send_message(session_id: str, message: str):
    """Send a user message"""
    
    if session_id not in sessions:
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found"}
        )
    
    # Get state
    state = sessions[session_id]
    
    # Process message
    result = medical_system.process_user_message(state, message)
    
    # Update session
    sessions[session_id] = result['state']
    
    return {
        "message": result['ai_message'],
        "collected_data": result['collected_data'],
        "is_complete": result['is_complete']
    }


@app.websocket("/ws/interview/{session_id}")
async def websocket_interview(websocket: WebSocket, session_id: str):
    """WebSocket for streaming responses"""
    await websocket.accept()
    
    # Initialize or get session
    if session_id not in sessions:
        result = medical_system.start_interview()
        sessions[session_id] = result['state']
        await websocket.send_json({
            "type": "message",
            "content": result['ai_message']
        })
    
    state = sessions[session_id]
    
    while True:
        # Receive user message
        data = await websocket.receive_json()
        user_message = data['message']
        
        # Stream response
        collected_text = ""
        async for chunk in medical_system.process_user_message_streaming(state, user_message):
            if 'content' in chunk:
                token = chunk['content']
                collected_text += token
                await websocket.send_json({
                    "type": "token",
                    "content": token
                })
        
        # Send completion
        await websocket.send_json({
            "type": "complete",
            "collected_data": state['collected_data'],
            "is_complete": state['all_sections_done']
        })
        
        # Update session
        sessions[session_id] = state


# ========== SUMMARY: How Files Work Together ==========

"""
FILE 1 (LangGraph Structure):
├─ Defines STATE (what data we track)
├─ Defines FLOW (how conversation moves)
├─ Defines NODES (what happens at each step)
└─ Defines ROUTING (where to go next)

FILE 2 (Urdu Configuration):
├─ Defines LANGUAGE rules (how to communicate)
├─ Defines PROMPTS (what to say in each section)
├─ Defines TERMS (medical vocabulary in Urdu)
└─ Defines CULTURAL context (Pakistani norms)

INTEGRATION (This File):
├─ Combines both files
├─ Agent node: Uses Urdu prompts + LangGraph flow
├─ Tool node: Structures Urdu responses into data
├─ Router: Controls flow based on conversation state
└─ API: Exposes to your frontend

FLOW:
User Message (Urdu) 
  → Agent Node (Uses Urdu prompt + LLM) 
  → LLM Response (Urdu) + Tool Calls
  → Tool Node (Structures data)
  → Router (Decides next step)
  → Next Section or Wait for User
"""


if __name__ == "__main__":
    main()