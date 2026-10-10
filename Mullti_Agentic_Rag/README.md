# Multi-Agentic RAG System

A sophisticated multi-agent RAG system with domain-specific knowledge bases and LangGraph orchestration.

## 🤖 Agent Architecture

### Agents

1. **Orchestrator Agent**
   - Routes queries to appropriate domains
   - Supports multi-domain queries
   - Uses LLM for classification

2. **Researcher Agent**
   - Retrieves from domain-specific indexes
   - Parallel retrieval across multiple domains
   - Returns top-k relevant documents per domain

3. **Analyst Agent**
   - Extracts key facts from retrieved context
   - Structures findings as bullet points
   - Focuses on query-relevant information

4. **Critic Agent**
   - Evaluates completeness of findings
   - Identifies gaps and missing information
   - Quality control checkpoint

5. **Synthesizer Agent**
   - Produces final coherent answer
   - Incorporates all findings and critique
   - Cites specific facts from context

## 🗂️ Domain Structure

### Knowledge Bases

| Domain | CSV File | Pinecone Index | Namespace | Dimension |
|--------|----------|----------------|-----------|-----------|
| Orders | `order.csv` | `single-agentic-rag` | `orders` | 384 |
| Products | `product.csv` | `single-agentic-rag` | `products` | 384 |
| Support | `support_catalogue.csv` | `single-agentic-rag` | `support` | 384 |

### Embeddings

Uses `sentence-transformers` (all-MiniLM-L6-v2) for:
- Consistent 384-dimensional embeddings
- Fast inference
- Good quality for semantic search

## 🚀 Quick Start

### Installation

```bash
cd Mullti_Agentic_Rag
uv sync
```

### Setup

1. **Create `.env` file** (in parent directory):
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_key
   PINECONE_API_KEY=your_pinecone_key
   ANTHROPIC_MODEL=claude-sonnet-4-6
   PINECONE_INDEX_NAME=single-agentic-rag
   ```

2. **Prepare knowledge bases**:
   - Ensure CSV files exist in `knowledgebase/` folder
   - Files: `order.csv`, `product.csv`, `support_catalogue.csv`

### Usage

#### First Run: Ingest Data

```bash
$env:INGEST="true"
uv run .\multi_agent_rag.py
```

This creates the shared Pinecone index if needed and ingests each CSV into its domain namespace.

#### Subsequent Runs: Query

```bash
uv run .\multi_agent_rag.py
```

#### Custom Queries

```python
from multi_agent_rag import MultiAgentRAG

# Initialize
rag = MultiAgentRAG()

# Query
result = rag.invoke("Where is my order #1234?")

# View results
print(result['answer'])
print(result['findings'])
print(result['critique'])
```

## 📊 Workflow

```
Query
  ↓
Orchestrator (classifies domain: orders/products/support)
  ↓
Researcher (retrieves from domain-specific indexes)
  ↓
Analyst (extracts key findings)
  ↓
Critic (checks completeness)
  ↓
Synthesizer (produces final answer)
  ↓
Answer
```

## 🔧 Configuration

### Index Configuration

Edit `pinecone_multi_index.py`:

```python
DEFAULT_INDEX_CONFIGS = [
    {"name": "single-agentic-rag", "dimension": 384, "metric": "cosine"},
]
```

### LLM Configuration

Edit `multi_agent_rag.py`:

```python
def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0.3,  # Adjust for creativity vs. determinism
    )
```

## 📁 Files

- `multi_agent_rag.py` - Main multi-agent pipeline
- `pinecone_multi_index.py` - Index management utilities
- `embeddings_sentance.py` - Sentence-transformer embeddings
- `rag_with_langgraph.py` - Alternative LangGraph implementation
- `knowledgebase/` - CSV data files
  - `order.csv` - Order information
  - `product.csv` - Product catalog
  - `support_catalogue.csv` - Support articles
- `pyproject.toml` - Dependencies

## 🧪 Example Queries

### Orders Domain
```python
"Where is my order #1234 and expected delivery date?"
"Track order ABC123"
"When will my order arrive?"
```

### Products Domain
```python
"List features of product ABC-900 and its price"
"What products do you have in stock?"
"Compare product X and Y"
```

### Support Domain
```python
"How to reset my password and contact support?"
"Troubleshoot login issues"
"What are your business hours?"
```

### Multi-Domain
```python
"I ordered product X, where is it and how do I return it?"
# Routes to: orders + products + support
```

## 🔍 Monitoring

### View Agent Flow

The system prints detailed logs:

```
Q: Where is my order #1234?
--- Findings ---
• Order #1234 shipped on Jan 15
• Expected delivery: Jan 20
• Tracking: XYZ789
--- Critique ---
All information present. Status is clear.
--- Answer ---
Your order #1234 shipped on January 15...
```

### Index Stats

```python
from pinecone_multi_index import PineconeIndexManager

mgr = PineconeIndexManager()
# View index statistics
```

## 🎯 Use Cases

- **E-commerce Support**: Handle orders, products, and support queries
- **Multi-Department Systems**: Route to specialized knowledge bases
- **Enterprise Knowledge Management**: Domain-specific RAG at scale
- **Customer Service Automation**: Intelligent query routing

## 🔄 Extending

### Add New Domain

1. **Create CSV file**: `knowledgebase/new_domain.csv`

2. **Add index config** in `pinecone_multi_index.py`:
   ```python
   {"name": "new-domain-index", "dimension": 384, "metric": "cosine"}
   ```

3. **Update ingestion** in `multi_agent_rag.py`:
   ```python
   ("knowledgebase/new_domain.csv", "new-domain-index")
   ```

4. **Update orchestrator** to recognize new domain:
   ```python
   # In orchestrate() method
   "domains": ["orders", "products", "support", "new_domain"]
   ```

### Use Different Embeddings

1. Modify `embeddings_sentance.py` to use different model
2. Update `dimension` in index configs
3. Re-ingest data with `INGEST=true`

## 📊 Performance

### Retrieval Speed
- Parallel retrieval across domains
- Top-k=5 per domain by default
- ~500ms average query time

### Accuracy
- Multi-agent critique improves accuracy
- Domain routing reduces noise
- Structured findings for clarity

## 🐛 Troubleshooting

**Issue**: Indexes not found
- **Fix**: Run with `INGEST=true` first

**Issue**: Wrong domain routing
- **Fix**: Improve orchestrator prompt or adjust temperature

**Issue**: CSV not found
- **Fix**: Verify files exist in `knowledgebase/` folder

**Issue**: Dimension mismatch
- **Fix**: Ensure embeddings match index dimension (384)

## 📚 Dependencies

- `langchain` - LangChain framework
- `langchain-anthropic` - Claude LLM
- `langchain-huggingface` - Hugging Face embedding integration
- `langchain-community` - Community components
- `langchain-pinecone` - Pinecone integration
- `langchain-text-splitters` - Text splitting
- `langgraph` - Graph orchestration
- `pinecone` - Vector database
- `sentence-transformers` - Embeddings
- `python-dotenv` - Environment variables

## 🎓 Advanced Topics

### LangGraph State Management

```python
class State:
    question: str
    domains: List[str]
    retrieved_docs: List
    findings: str
    critique: str
    answer: str
```

### Conditional Routing

The graph uses conditional edges based on:
- Orchestrator's domain classification
- Critic's assessment of completeness

### Parallel Retrieval

Researcher retrieves from multiple indexes simultaneously:
```python
for domain in domains:
    docs = retriever.invoke(question)
```

## 🔗 Related Projects

- **Modular_Rag**: Single-agent modular implementation
- **CRAG**: Single-agent with corrective mechanisms
- **Single_agentic-rag**: Single-agent with clean architecture

---

**Multi-Domain • Multi-Agent • Production-Scale**
