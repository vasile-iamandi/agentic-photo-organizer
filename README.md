[Screencast from 2026-11-14 23-29-16.webm](https://github.com/user-attachments/assets/3ee95f7d-ea14-4379-932a-9f98f469dce5)


# Immich LangChain Assistant with Human-in-the-Loop (HITL)

An intelligent conversational agent and interactive web application powered by **LangChain**, **LangGraph**, and **Azure OpenAI**, integrated with a self-hosted **Immich** photo server.

The assistant allows users to search photos in natural language (English or other languages), automatically create and manage albums, view interactive galleries directly in the chat, and supervise sensitive operations through Human-in-the-Loop confirmation checkpoints.

---

## 📌 Problem

Self-hosted photo management solutions like **Immich** offer powerful machine learning capabilities (such as CLIP-based semantic vector search, facial recognition, and metadata indexing) via REST APIs. However:

1. **Lack of Conversational Interface**: Users traditionally must navigate fixed UI filters or exact search bars rather than describing what they want naturally (e.g., *"Find the photos from our mountain vacation and create a new album with the best ones"*).

2. **Language Barrier in CLIP Models**: Open-source CLIP embedding models frequently yield significantly higher accuracy with English prompts than with other languages.

3. **Risk of Unintended Side Effects**: Allowing an autonomous AI agent to create, modify, or organize photos directly in a personal photo library risks altering albums incorrectly without user oversight.

---

## 🏛️ Architecture & Workflows

The application coordinates an interactive **Gradio** web chat, a cyclic state graph in **LangGraph**, **Azure OpenAI** for reasoning and multilingual translation, and a local **Immich** server.

Detailed diagram sources are also available in [`architecture/`](./architecture):
- **PlantUML**: [`architecture/architecture.puml`](./architecture/architecture.puml) and [`architecture/message_flow.puml`](./architecture/message_flow.puml)

### Key Workflow Components:
1. **Agent State & Checkpointing**: Powered by `langgraph` and `MemorySaver` to preserve multi-turn conversational context while isolating newly found assets per turn.
2. **Dynamic Routing**: Separates read-only tools (`search_immich_assets`, `get_immich_asset`, `get_immich_albums`) from mutating operations.
3. **HITL Interruption**: Suspends graph execution with native `interrupt()` before executing album creations (`create_immich_album`, `add_assets_to_immich_album`), presenting the user with photo previews and approval controls.

---

## 🛠️ Technologies

- **Core Agent & Orchestration**: [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/) (Stateful cyclic workflows, Checkpointing, Tool Calling).
- **LLM Provider**: [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) (`langchain-openai` / `AzureChatOpenAI`).
- **Web UI & Streaming**: [Gradio](https://gradio.app/) (Responsive chat UI, Custom CSS, Native `gr.Gallery` rendering, streaming async generators).
- **API Client & Networking**: [HTTPX](https://www.python-httpx.org/) (Async/sync REST client with timeout protection).
- **Data Validation & Typing**: [Pydantic v2](https://docs.pydantic.dev/).
- **Package Management**: [uv](https://github.com/astral-sh/uv) / Python 3.12+.

---

## 🚀 How to Run

### 1. Prerequisites & Immich Setup
- Python `>= 3.12`
- [uv](https://docs.astral.sh/uv/) package manager installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker & Docker Compose for running Immich locally
- Azure OpenAI resource (Endpoint, API Key, and Model Deployment)

#### Setting up Immich via Docker Compose:
The repository provides a complete Docker Compose environment located in [`immich_setup/`](./immich_setup):

1. **Configure Immich environment**:
   Edit `immich_setup/.env` to adjust storage paths, database credentials, or ports:
   ```bash
   cd immich_setup
   # Edit .env (UPLOAD_LOCATION, DB_DATA_LOCATION, DB_PASSWORD, etc.)
   ```

2. **Start the Immich stack**:
   ```bash
   docker compose up -d
   ```
   This launches `immich_server` (port `2283`), `immich_machine_learning` (CLIP embeddings for smart search), `immich_postgres` (with vector support), and `immich_redis`.

3. **Obtain your Immich API Key**:
   - Open your browser at `http://localhost:2283` and complete initial admin onboarding (or login).
   - Navigate to **Account Settings** $\rightarrow$ **API Keys** $\rightarrow$ **New API Key**.
   - Copy the generated API key.

#### 🤖 Setting up Azure OpenAI:
1. **Create an Azure OpenAI Resource** in the [Azure Portal](https://portal.azure.com/) or [Azure AI Foundry / Studio](https://ai.azure.com/).
2. **Deploy a Model**: e.g., `gpt-4.1-nano`, `gpt-4o`, or `gpt-4o-mini`.
3. **Retrieve Endpoint & API Key**:
   - Note the **Endpoint** (`https://<resource-name>.openai.azure.com`).
   - Copy **Key 1** (or **Key 2**).

---

### 2. Environment Configuration

Create or update `application/.env`:

```env
# Immich Server Configuration
IMMICH_SERVER_URL=http://localhost:2283
IMMICH_API_KEY=your_immich_api_key_here

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-nano
AZURE_OPENAI_MODEL_NAME=gpt-4.1-nano
```

---

### 3. Install Dependencies & Launch

Navigate into the `application` directory:

```bash
cd application

# Install dependencies
uv sync

# Launch the Gradio web application
uv run application.py
# or using Makefile:
make run
```

Open your browser and navigate to: `http://localhost:7860`

---

## 🧠 AI Component

### 1. Conversational Reasoning & Tool Calling
The system utilizes Azure OpenAI models with structured function calling. The model dynamically inspects available tools and generates accurate JSON payloads for:

- `search_immich_assets(query, size, page)`: Smart CLIP semantic search for photos.
- `create_immich_album(album_name, description, asset_ids)`: Creates albums and attaches photos.
- `get_immich_albums()`: Lists all existing albums.
- `add_assets_to_immich_album(album_id, asset_ids)`: Appends assets to an album.
- `get_immich_asset(asset_id)`: Fetches metadata for a specific photo.

### 2. Multilingual-to-English Semantic Translation
Immich's Smart Search relies on CLIP embeddings trained predominantly on English datasets. The agent's system prompt enforces automatic semantic translation of search concepts (e.g., user asks for *"câini"* $\rightarrow$ agent translates to `query="dog"`), avoiding false negatives and irrelevant search results.

### 3. Multi-Step Execution
For composite queries like *"fă un album separat și pune o poză cu pisica"*:
1. The agent first calls `search_immich_assets(query="cat")`.
2. Inspects candidate assets and extracts the best matching ID.
3. Proposes `create_immich_album` with the extracted asset ID.
4. Triggers `interrupt()` for human confirmation before committing changes to Immich.

---

## 📐 Design Decisions

1. **Strict Separation of Concerns**:
   - [`application/immich_tools.py`](./application/immich_tools.py): Low-level HTTP client and typed tool definitions.
   - [`application/graph.py`](./application/graph.py): LangGraph state graph, checkpointing, routing, and tool calling logic.
   - [`application/immich_agent.py`](./application/immich_agent.py): High-level wrapper handling streaming events, interrupts, and token generation.
   - [`application/application.py`](./application/application.py): Gradio presentation layer with interactive gallery and Human-in-the-Loop decision controls.

2. **Human-in-the-Loop via Native LangGraph `interrupt()`**:
   - Mutating tools are flagged as `SENSITIVE_TOOLS` (`create_immich_album`, `add_assets_to_immich_album`).
   - The graph suspends execution and yields control to the UI.
   - On approval, the graph resumes seamlessly; on rejection, a cancellation message is injected so the LLM acknowledges the refusal politely.

3. **Per-Turn Asset Extraction**:
   - Multi-turn chats accumulate previous tool outputs. To prevent showing duplicate images from previous queries, asset extraction filters messages strictly within the active turn.

4. **Responsive Gallery & Previews**:
   - Rather than returning raw URLs, thumbnails are fetched using authenticated Immich endpoints and rendered using Gradio's native `gr.Gallery` with full-screen preview support.
