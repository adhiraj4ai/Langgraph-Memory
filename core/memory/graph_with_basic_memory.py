from typing import List

import tiktoken
import decouple
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from core.memory.memory_manager import BasicMemory
from core.vendor.factory import llm_agent

tokenizer = tiktoken.encoding_for_model(decouple.config("MODEL"))

class GraphWithBasicMemory:
    # Define the prompt template for the agent
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant with advanced long-term memory capabilities. Powered by a stateless LLM, "
                "you must rely on external memory to store information between conversations. Utilize the available "
                "memory tools to store and retrieve important details that will help you better attend to the user's"
                " needs and understand their context.\n\n"
                "Memory Usage Guidelines:\n"
                "1. Actively use memory tools (save_core_memory, save_recall_memory) to build a comprehensive "
                "understanding of the user.\n"
                "2. Make informed suppositions and extrapolations based on stored memories.\n"
                "3. Regularly reflect on past interactions to identify patterns and preferences.\n"
                "4. Update your mental model of the user with each new piece of information.\n"
                "5. Cross-reference new information with existing memories for consistency.\n"
                "6. Prioritize storing emotional context and personal values alongside facts.\n"
                "7. Use memory to anticipate needs and tailor responses to the user's style.\n"
                "8. Recognize and acknowledge changes in the user's situation or perspectives over time.\n"
                "9. Leverage memories to provide personalized examples and analogies.\n"
                "10. Recall past challenges or successes to inform current problem-solving.\n\n"
                "## Recall Memories\n"
                "Recall memories are contextually retrieved based on the current conversation:\n{recall_memories}\n\n"
                "## Instructions\n"
                " Engage with the user naturally, as a trusted colleague or friend."
                " There's no need to explicitly mention your memory capabilities."
                " Instead, seamlessly incorporate your understanding of the user"
                " into your responses. Be attentive to subtle cues and underlying"
                " emotions. Adapt your communication style to match the user's"
                " preferences and current emotional state. Use tools to persist"
                " information you want to retain in the next conversation. If you"
                " do call tools, all text preceding the tool call is an internal"
                " message. Respond AFTER calling the tool, once you have"
                " confirmation that the tool completed successfully.\n\n",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    memory = BasicMemory()

    class MemoryState(MessagesState):
        recall_memories: List[str]

    def __init__(self):
        self.model = llm_agent.get_llm()
        self.tools = [self.memory.save_memory, self.memory.search_memories]
        self.model_with_tools = self.model.bind_tools(self.tools)

    def llm_agent_with_memory(self, state: MemoryState) -> MemoryState:
        bound = self.prompt | self.model_with_tools
        recall_str = (
                "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
        )
        prediction = bound.invoke(
            {
                "messages": state["messages"],
                "recall_memories": recall_str,
            }
        )
        return {
            "messages": [prediction],
        }

    @staticmethod
    def load_memories(state: MemoryState, config: RunnableConfig) -> MemoryState:
        convo_str = get_buffer_string(state["messages"])
        convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
        recall_memories = GraphWithBasicMemory.memory.search_memories.invoke(convo_str, config)
        return {
            "recall_memories": recall_memories,
        }

    @staticmethod
    def route_tools(state: MemoryState):
        """Determine whether to use tools or end the conversation based on the last message.

        Args:
            state (schemas.State): The current state of the conversation.

        Returns:
            Literal["tools", "__end__"]: The next step in the graph.
        """
        msg = state["messages"][-1]
        if msg.tool_calls:
            return "tools"

        return END

    def build(self):
        # Create the graph and add nodes
        builder = StateGraph(self.MemoryState)
        builder.add_node("load_memories", self.load_memories)
        builder.add_node("agent", self.llm_agent_with_memory)
        builder.add_node("tools", ToolNode(self.tools))

        # Add edges to the graph
        builder.add_edge(START, "load_memories")
        builder.add_edge("load_memories", "agent")
        builder.add_conditional_edges("agent", self.route_tools, ["tools", END])
        builder.add_edge("tools", "agent")

        # Compile the graph
        memory = MemorySaver()
        self.graph = builder.compile(checkpointer=memory)
