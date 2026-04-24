"""
10 multi-turn benchmark conversations.
Each Conversation tests a specific memory capability.
Turns with expects_memory=True are "checkpoints" where
the agent MUST correctly recall specified information.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Turn:
    user:           str
    expects_memory: bool = False   # Is this a memory-recall checkpoint?
    memory_hint:    str  = ""      # What the agent should recall


@dataclass
class Conversation:
    id:                  int
    name:                str
    description:         str
    memory_type_tested:  str       # preference / episodic / semantic / context_trim
    turns:               List[Turn] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
CONVERSATIONS: List[Conversation] = [

    # 1. User Preference Tracking
    Conversation(
        id=1,
        name="User Preference Tracking",
        description="User states preferences early; agent applies them in later turns.",
        memory_type_tested="preference",
        turns=[
            Turn("My name is Alex. I'm a software engineer who loves Python."),
            Turn("I prefer detailed technical answers with code examples."),
            Turn(
                "What's the difference between a Python list and a tuple?",
                expects_memory=True,
                memory_hint="User is a Python engineer; prefers detailed technical answers with code",
            ),
            Turn("I'm also working on an ML project with scikit-learn."),
            Turn(
                "Recommend an algorithm for a multi-class classification problem.",
                expects_memory=True,
                memory_hint="User likes Python/ML + wants detailed technical answers",
            ),
        ],
    ),

    # 2. Multi-Session Continuity (long-term Redis test)
    Conversation(
        id=2,
        name="Multi-Session Continuity",
        description="Long-term memory survives a simulated session restart.",
        memory_type_tested="long_term",
        turns=[
            Turn("I'm learning Vietnamese and I find tones really challenging."),
            Turn("My favorite Vietnamese food is phở bò."),
            Turn("--- SESSION RESTART ---"),      # handled by benchmark runner
            Turn(
                "Hello again! Can you recommend some Vietnamese dishes for me?",
                expects_memory=True,
                memory_hint="User's favorite Vietnamese dish is phở bò",
            ),
            Turn(
                "Any tips for picking up Vietnamese faster?",
                expects_memory=True,
                memory_hint="User said they find Vietnamese tones challenging",
            ),
        ],
    ),

    # 3. Experience Recall / Cross-Turn Reference
    Conversation(
        id=3,
        name="Experience Recall — Cross-Turn Reference",
        description="Agent must recall content from 5+ turns back.",
        memory_type_tested="episodic",
        turns=[
            Turn("What's the most efficient type of solar panel available today?"),
            Turn("How does wind energy compare in efficiency to solar?"),
            Turn("What about nuclear power as a clean energy source?"),
            Turn("What are the economic costs of transitioning to 100% renewables?"),
            Turn(
                "Going back to what you said about solar panels — which manufacturers would you recommend now?",
                expects_memory=True,
                memory_hint="Turn 1 discussed solar panel efficiency types",
            ),
        ],
    ),

    # 4. Semantic Similarity Recall
    Conversation(
        id=4,
        name="Semantic Similarity Recall",
        description="Agent retrieves similar memory even when query is differently worded.",
        memory_type_tested="semantic",
        turns=[
            Turn("What are the health benefits of drinking green tea every day?"),
            Turn("Is coffee better or worse for health compared to tea?"),
            Turn("How much water should a healthy adult drink daily?"),
            Turn(
                "What are the advantages of consuming matcha on a regular basis?",
                expects_memory=True,
                memory_hint="Turn 1 already discussed green tea/matcha health benefits",
            ),
        ],
    ),

    # 5. Preference Override
    Conversation(
        id=5,
        name="Preference Override",
        description="User changes preference mid-conversation; agent must adapt immediately.",
        memory_type_tested="preference",
        turns=[
            Turn("Please keep your answers very concise — 2 sentences max."),
            Turn(
                "What is machine learning?",
                expects_memory=True,
                memory_hint="User wants ≤2 sentence answers",
            ),
            Turn("Actually, I changed my mind. I want very long, detailed explanations now."),
            Turn(
                "What is deep learning?",
                expects_memory=True,
                memory_hint="User updated preference: wants DETAILED/LONG explanations",
            ),
        ],
    ),

    # 6. Factual Knowledge Retention
    Conversation(
        id=6,
        name="Factual Knowledge Retention",
        description="Agent learns facts about the user's company and recalls them accurately.",
        memory_type_tested="factual",
        turns=[
            Turn("Our company is called TechViet and we have 250 employees."),
            Turn("We're headquartered in Hanoi and specialise in fintech."),
            Turn("Our flagship product is PayBridge — a cross-border payment platform."),
            Turn(
                "Give me a quick company overview.",
                expects_memory=True,
                memory_hint="Must include: TechViet, 250 employees, Hanoi, fintech, PayBridge",
            ),
        ],
    ),

    # 7. Long Conversation — Context Trim Test
    Conversation(
        id=7,
        name="Long Conversation — Context Trim Resilience",
        description="Critical info from turn 1 must survive 17 filler turns (stress-tests P1/P2 eviction).",
        memory_type_tested="context_trim",
        turns=[
            Turn("My name is Mai and my employee ID is EMP-7890."),   # Critical early fact
            *[Turn(f"Tell me one interesting fact about topic {i}.") for i in range(1, 18)],
            Turn(
                "What is my name and employee ID?",
                expects_memory=True,
                memory_hint="Name=Mai, Employee ID=EMP-7890 stated in turn 1",
            ),
        ],
    ),

    # 8. Temporal Reasoning
    Conversation(
        id=8,
        name="Temporal Reasoning",
        description="Agent must identify and recall turns by their position in the conversation.",
        memory_type_tested="episodic",
        turns=[
            Turn("What is the capital of France?"),
            Turn("What currency is used in Japan?"),
            Turn("How far is the Moon from Earth?"),
            Turn(
                "What was the very first question I asked you in this conversation?",
                expects_memory=True,
                memory_hint="First question was about the capital of France",
            ),
            Turn(
                "What did I ask you right before this message?",
                expects_memory=True,
                memory_hint="Previous question was about recalling the first question",
            ),
        ],
    ),

    # 9. Allergy Conflict Update (MANDATORY rubric test)
    Conversation(
        id=9,
        name="Allergy Conflict Update (Mandatory Test)",
        description="User corrects a fact; agent must OVERWRITE, not append. Rubric test bat buoc.",
        memory_type_tested="conflict",
        turns=[
            Turn("My name is Linh and I work as a data analyst."),
            Turn("I'm allergic to cow's milk."),
            Turn(
                "What foods should I avoid given my allergy?",
                expects_memory=True,
                memory_hint="User is allergic to cow's milk",
            ),
            Turn("Oh wait, I made a mistake. I'm actually allergic to soybeans, not cow's milk."),
            Turn(
                "Please update my profile. What's my allergy now?",
                expects_memory=True,
                memory_hint="Allergy corrected: soybeans (NOT cow's milk). Profile must show allergy=soybeans",
            ),
            Turn(
                "Recommend some safe snacks for me based on my allergy profile.",
                expects_memory=True,
                memory_hint="User allergy is soybeans. Must NOT mention cow's milk allergy",
            ),
        ],
    ),


    # 10. Multi-Topic Context Switching
    Conversation(
        id=10,
        name="Multi-Topic Context Switching",
        description="Topics switch back and forth; agent must track all threads simultaneously.",
        memory_type_tested="context",
        turns=[
            Turn("I'm planning a trip to Japan next month."),
            Turn("What programming language should I learn first for web development?"),
            Turn(
                "Back to Japan — what are the must-see places in Tokyo?",
                expects_memory=True,
                memory_hint="User is planning a Japan trip (turn 1)",
            ),
            Turn("I've decided to learn JavaScript for web dev."),
            Turn(
                "For my Japan trip, when is the best time to visit Kyoto?",
                expects_memory=True,
                memory_hint="User planning Japan trip + decided to learn JavaScript",
            ),
        ],
    ),
]
