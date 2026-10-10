"""
Multi-Agent RAG with LangGraph over Pinecone domain namespaces

Agents:
 - Orchestrator: routes query to domain(s): orders, products, support
 - Researcher: retrieves from appropriate retriever(s)
 - Analyst: synthesizes retrieved snippets into structured findings
 - Critic: checks gaps/consistency and requests follow-up retrieval if needed
 - Synthesizer: produces final answer

Vector store: one Pinecone index with a namespace per domain
LLM: Anthropic Claude
"""
import os
import csv
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END

from pinecone_multi_index import PineconeIndexManager, DEFAULT_INDEX_CONFIGS


load_dotenv()


RAG_INDEX = os.getenv("PINECONE_INDEX_NAME", "single-agentic-rag")
ORDERS_NAMESPACE = "orders"
PRODUCTS_NAMESPACE = "products"
SUPPORT_NAMESPACE = "support"


class RAGState(TypedDict, total=False):
    question: str
    domains: List[str]
    retrieved_docs: List[Document]
    context: str
    findings: str
    critique: str
    answer: str


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.3,
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ensure_indexes() -> None:
    mgr = PineconeIndexManager()
    mgr.ensure_indexes(DEFAULT_INDEX_CONFIGS)


def load_csv_documents(csv_path: str) -> List[Document]:
    """Load CSV rows as LangChain documents without deprecated loaders."""
    documents = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row_number, row in enumerate(reader, start=2):
            content = "\n".join(
                f"{column.replace('_', ' ')}: {value}"
                for column, value in row.items()
                if column and value not in (None, "")
            )
            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": csv_path, "row": row_number},
                    )
                )
    return documents


def ingest_csvs(knowledge_dir: str = "knowledgebase") -> None:
    """Ingest CSVs into their respective Pinecone indexes.

    - order.csv -> orders namespace
    - product.csv -> products namespace
    - support_catalogue.csv -> support namespace
    """
    ensure_indexes()

    embeddings = get_embeddings()

    mapping = [
        (os.path.join(knowledge_dir, "order.csv"), ORDERS_NAMESPACE),
        (os.path.join(knowledge_dir, "product.csv"), PRODUCTS_NAMESPACE),
        (os.path.join(knowledge_dir, "support_catalogue.csv"), SUPPORT_NAMESPACE),
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for csv_path, namespace in mapping:
        if not os.path.exists(csv_path):
            print(f"⚠️ Missing CSV: {csv_path}")
            continue
        print(f"📥 Ingesting {csv_path} -> {RAG_INDEX}/{namespace}")
        docs = load_csv_documents(csv_path)
        chunks = splitter.split_documents(docs)

        # Upsert into Pinecone via LangChain vectorstore
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=RAG_INDEX,
            namespace=namespace,
        )
        print(f"✅ Ingested {len(chunks)} chunks into namespace '{namespace}'")


class MultiAgentRAG:
    def __init__(self):
        self.llm = get_llm()
        self.embeddings = get_embeddings()
        ensure_indexes()

        # Build retrievers for each index
        self.orders_vs = PineconeVectorStore(
            index_name=RAG_INDEX,
            namespace=ORDERS_NAMESPACE,
            embedding=self.embeddings,
        )
        self.products_vs = PineconeVectorStore(
            index_name=RAG_INDEX,
            namespace=PRODUCTS_NAMESPACE,
            embedding=self.embeddings,
        )
        self.support_vs = PineconeVectorStore(
            index_name=RAG_INDEX,
            namespace=SUPPORT_NAMESPACE,
            embedding=self.embeddings,
        )

        self.orders_ret = self.orders_vs.as_retriever(search_kwargs={"k": 5})
        self.products_ret = self.products_vs.as_retriever(search_kwargs={"k": 5})
        self.support_ret = self.support_vs.as_retriever(search_kwargs={"k": 5})

        self.graph = self._build_graph()

    # ============ Nodes ============
    def orchestrate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        system = (
            "You are the Orchestrator. Classify the user question into one or more of "
            "these domains: orders, products, support. Output a JSON with key 'domains' "
            "as an array of domains. If unclear, pick the most likely one."
        )
        prompt = f"{system}\n\nQuestion: {question}\nRespond with JSON only."
        resp = self.llm.invoke(prompt)

        domains: List[str] = []
        try:
            import json
            data = json.loads(resp.content)
            if isinstance(data, dict) and isinstance(data.get("domains"), list):
                domains = [d for d in data["domains"] if d in ["orders", "products", "support"]]
        except Exception:
            pass

        if not domains:
            domains = ["products"]
        return {"domains": domains}

    def research(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        domains: List[str] = state.get("domains", [])
        all_docs = []

        for domain in domains:
            if domain == "orders":
                docs = self.orders_ret.invoke(question)
            elif domain == "products":
                docs = self.products_ret.invoke(question)
            else:
                docs = self.support_ret.invoke(question)
            all_docs.extend(docs)

        context = "\n\n".join(d.page_content for d in all_docs)
        return {"retrieved_docs": all_docs, "context": context}

    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        context = state.get("context", "")
        sys = (
            "You are the Analyst. Read the retrieved context and extract key facts "
            "relevant to the question. Return a crisp bullet list of findings."
        )
        prompt = f"{sys}\n\nQuestion: {question}\nContext:\n{context}\n\nFindings:"
        resp = self.llm.invoke(prompt)
        return {"findings": resp.content}

    def critique(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        findings = state.get("findings", "")
        sys = (
            "You are the Critic. Assess if findings answer the question fully. "
            "Point out missing info concisely. If sufficient, say 'OK'."
        )
        prompt = f"{sys}\n\nQuestion: {question}\nFindings:\n{findings}\n\nCritique:"
        resp = self.llm.invoke(prompt)
        return {"critique": resp.content}

    def synthesize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        context = state.get("context", "")
        findings = state.get("findings", "")
        critique = state.get("critique", "")
        sys = (
            "You are the final Synthesizer. Using the context, findings, and critique, "
            "produce a clear, helpful answer. Cite specific facts from context when useful."
        )
        prompt = (
            f"{sys}\n\nQuestion: {question}\nContext:\n{context}\n\n"
            f"Findings:\n{findings}\n\nCritique:\n{critique}\n\nFinal Answer:"
        )
        resp = self.llm.invoke(prompt)
        return {"answer": resp.content}

    # ============ Graph ============
    def _build_graph(self):
        g = StateGraph(RAGState)
        g.add_node("orchestrator", self.orchestrate)
        g.add_node("researcher", self.research)
        g.add_node("analyst", self.analyze)
        g.add_node("critic", self.critique)
        g.add_node("synthesizer", self.synthesize)

        g.set_entry_point("orchestrator")
        g.add_edge("orchestrator", "researcher")
        g.add_edge("researcher", "analyst")
        g.add_edge("analyst", "critic")
        g.add_edge("critic", "synthesizer")
        g.add_edge("synthesizer", END)
        return g.compile()

    def invoke(self, question: str) -> Dict[str, Any]:
        result = self.graph.invoke({"question": question})
        return result


def main():
    # Optional: initial ingest if needed
    if os.getenv("INGEST", "false").lower() in {"1", "true", "yes"}:
        ingest_csvs()

    rag = MultiAgentRAG()
    queries = [
        "What is the status of order 5001?",
        "What are the features and price of the Wireless Mouse?",
        "How do I return a product?",
    ]
    for q in queries:
        print("\n" + "=" * 60)
        print(f"Q: {q}")
        out = rag.invoke(q)
        print("--- Findings ---")
        print(out.get("findings", ""))
        print("--- Critique ---")
        print(out.get("critique", ""))
        print("--- Answer ---")
        print(out.get("answer", ""))


if __name__ == "__main__":
    main()


