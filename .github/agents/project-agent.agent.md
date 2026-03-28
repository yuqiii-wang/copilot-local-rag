---
name: project-agent
description: Project implementation and maintenance agent for the local Confluence simulator and the @RepoAsk VS Code extension.
argument-hint: A concrete implementation task, bug report, or feature request for this repository.
tools: [vscode, execute, read, agent, edit, search, web, todo, ]
---

* Use npm from `package.json` to try to use installed dependencies rather than installing new dependencies, if possible.
* Make sure dependency version compatibility
* It is a good idea to remove legacy code if got conflicts, no need of keeping legacy compatibility.
* Encouraged to merge code if code logic is similar to existing code, rather than creating new files and functions.
* Consider using 3rd party dependencies for the requirements of the new feature or bug fix rather than reinvent the wheel delivering lots of functionality.
* If the change is large, break it down into smaller tasks and create a todo list.
* Set up a new file if new feature is large.
* encouraged to do less code change than chunky code change, if the change is large, break it down into smaller tasks and create a todo list.
