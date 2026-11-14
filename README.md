# Immich LangChain Assistant with Human-in-the-Loop (HITL)

An intelligent conversational agent and interactive web application powered by **LangChain**, **LangGraph**, and **Azure OpenAI**, integrated with a self-hosted **Immich** photo server.

The assistant allows users to search photos in natural language (Other languages/English), automatically create and manage albums, view interactive galleries directly in the chat, and supervise sensitive operations through Human-in-the-Loop confirmation checkpoints.

---

## 🚀 How to Run

### 1. Prerequisites & Immich Setup
- Python `>= 3.12`
- [uv](https://docs.astral.sh/uv/) package manager installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker & Docker Compose (for running Immich locally)

#### Setting up Immich via Docker Compose:
The repository provides a complete Docker Compose environment located in the [`immich_setup/`](./immich_setup) directory:

#### 1. Configure Immich environment:
In `immich_setup/.env`, adjust your storage paths, database credentials, and ports if needed:
   ```bash
   cd ../immich_setup # Edit .env (UPLOAD_LOCATION, DB_DATA_LOCATION, DB_PASSWORD, etc.)
   ```
#### 2. Start the Immich stack:
   ```bash
   docker compose up -d
   ```
   This launches `immich_server` (port `2283`), `immich_machine_learning` (CLIP embeddings for smart search), `immich_postgres` (with vector support), and `immich_redis`.
#### 3. Obtain your Immich API Key:
   - Open your browser at `http://localhost:2283` and complete initial admin onboarding (or login).
   - Navigate to **Account Settings** -> **API Keys** -> **New API Key**.
   - Copy the generated API key into your `application/.env` as `IMMICH_API_KEY`.

