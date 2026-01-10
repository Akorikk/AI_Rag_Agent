echo "Creating project structure inside existing ai-agent-rag repo..."

# Create directories
mkdir -p app
mkdir -p data/documents

# Create files inside app/
touch app/main.py
touch app/agent.py
touch app/rag.py
touch app/tools.py
touch app/memory.py

# Create root-level files if not present
touch requirements.txt
touch README.md

echo "Structure created successfully!"
