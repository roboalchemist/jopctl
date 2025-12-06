# jopctl

A CLI for the Joplin Data API with full CRUDL (create, read, update, delete, list) support.

## Installation

### Using uv (recommended)

```bash
uv tool install jopctl
```

### Using pip

```bash
pip install jopctl
```

### Using Homebrew

```bash
brew tap roboalchemist/tap
brew install jopctl
```

### From source

```bash
git clone https://github.com/roboalchemist/jopctl.git
cd jopctl
pip install .
```

## Setup

1. **Enable Joplin Web Clipper:**
   - Open Joplin desktop app
   - Go to Tools > Options > Web Clipper
   - Enable the Web Clipper service
   - Copy the authorization token

2. **Set environment variables:**
   ```bash
   export JOPLIN_API_PORT=41184
   export JOPLIN_API_TOKEN=<your-token>
   ```

3. **Verify connection:**
   ```bash
   jopctl auth status
   ```

## Usage

### Global Options

- `--port TEXT` - Joplin API port (default: 41184)
- `--token TEXT` - API token (required, or set JOPLIN_API_TOKEN)
- `--format [text|json|yaml]` - Output format (default: text)

### Commands

```bash
# Connection
jopctl auth status

# Notebooks
jopctl notebook list
jopctl notebook show "Projects"
jopctl notebook create "New Notebook" --parent "Projects"
jopctl notebook update "Old Name" --title "New Name"
jopctl notebook delete "Test Notebook"

# Notes
jopctl note list --notebook "Active" --limit 20
jopctl note show NOTE_ID --body
jopctl note create --title "New Note" --body "Content" --notebook "Ideas"
jopctl note create --title "From File" --body-file ./content.md
jopctl note update NOTE_ID --title "Updated" --body "New content"
jopctl note delete NOTE_ID [--permanent]

# Search
jopctl search "keyword" --type note --limit 10

# Tags
jopctl tag list
jopctl tag create "important"
jopctl tag attach TAG_ID NOTE_ID
jopctl tag detach TAG_ID NOTE_ID
jopctl tag notes TAG_ID
jopctl tag update TAG_ID "new-name"
jopctl tag delete TAG_ID

# Resources (attachments)
jopctl resource list --limit 10
jopctl resource show RESOURCE_ID
jopctl resource create --file ./image.png --title "Screenshot"
jopctl resource update RESOURCE_ID --title "New Title"
jopctl resource delete RESOURCE_ID

# Export
jopctl export ~/backup --export-format md
jopctl export ~/backup --export-format raw --resources
jopctl export ~/backup --notebook "Projects"
jopctl export ~/backup.jex --export-format jex
```

### Export Formats

Export your notes to various formats:

- **md** (default): Markdown files preserving folder structure
- **raw**: JSON files with full note metadata
- **jex**: Joplin Exchange format (tar archive)

```bash
# Export all notes as markdown
jopctl export ~/joplin-backup

# Export specific notebook with attachments
jopctl export ~/backup --notebook "Projects" --resources

# Export specific notes by ID
jopctl export ~/backup --notes "id1,id2,id3"

# Create portable Joplin archive
jopctl export ~/backup.jex --export-format jex --resources
```

### JSON Output for Scripting

Use `--format json` for programmatic parsing:

```bash
# Create and capture ID
NOTE_ID=$(jopctl --format json note create --title "Test" --notebook "Ideas" | jq -r '.id')

# Use ID for subsequent operations
jopctl note update "$NOTE_ID" --body "Updated content"
```

### Advanced Note Creation

Use `--data` to pass additional JSON fields:

```bash
# Create a todo with due date
jopctl note create --title "Task" --body "Do this" \
  --data '{"is_todo": 1, "todo_due": 1735689600000}'
```

## Requirements

- Python 3.9+
- Joplin desktop app with Web Clipper enabled

## License

MIT
