# Production Support Skill

## Overview
Comprehensive production support workflow for troubleshooting, incident response, and root cause analysis. Integrates local documentation, recent code changes, and team expertise for rapid problem resolution.

## Key Workflows

### 1. Local Documentation Search
Use RepoAsk to search your local knowledge base for relevant documentation:

```bash
/search confluence jira [your issue/query]
```

**Tags to focus on:**
- `confluence` - Internal documentation and runbooks
- `jira` - Issue tracking and known problems
- `incident` - Post-mortem reports and lessons learned
- `troubleshooting` - Common issue solutions

### 2. Recent Code Change Analysis

Automatically check for recent changes that may relate to the issue:

```bash
# For each project in workspace root:
cd {project-root}
git fetch origin
git log --oneline --merges --since="3 days ago" origin/main..HEAD
git log --oneline --merges --since="3 days ago" origin/master..HEAD
git diff origin/main...HEAD -- {affected-areas}
git diff origin/master...HEAD -- {affected-areas}
```

**What to check:**
- Dependency updates
- Configuration changes
- Error handling modifications
- Database schema changes
- API contract changes
- Performance optimizations

### 3. Workspace Repository Discovery

For workspaces with multiple projects:

```bash
# In VS Code workspace root
find . -maxdepth 2 -type d -name ".git" -exec dirname {} \;
```

This identifies all git repositories that need to be checked.
