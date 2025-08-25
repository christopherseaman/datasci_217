# MCP Server Troubleshooting

## Current Status

### Working:
- `claude-flow@alpha` package installed globally
- MCP servers running as processes:
  - `claude-flow@alpha mcp start` (PID 27153)
  - `ruv-swarm mcp start` (PID 26799)
- Project configuration files created

### Issues:
- Claude Code cannot access MCP tools (`mcp__claude-flow__*`, `mcp__ruv-swarm__*`)
- `claude mcp` commands not working (Claude CLI path issues)

### Configuration Files Status:

#### `.mcp.json` (Project Level):
```json
{
  "mcpServers": {
    "claude-flow": {
      "command": "npx",
      "args": ["claude-flow@alpha", "mcp", "start"],
      "type": "stdio"
    },
    "ruv-swarm": {
      "command": "npx", 
      "args": ["ruv-swarm@latest", "mcp", "start"],
      "type": "stdio"
    }
  }
}
```

#### `.claude/settings.json` (Project Level):
- Has hooks configuration
- Lists MCP tools in permissions: `mcp__ruv-swarm`, `mcp__claude-flow`

### Root Cause:
The MCP servers are running as standalone processes but Claude Code isn't connecting to them. This is likely because:

1. Claude Code expects MCP servers to be registered in global Claude configuration
2. The current approach runs servers independently without proper Claude integration
3. The `claude mcp` commands require proper Claude CLI setup

### Current Workaround:
Use claude-flow functionality through direct npx commands:
```bash
npx claude-flow@alpha sparc modes
npx claude-flow@alpha swarm "task description"
```

### Next Steps to Fix:
1. Properly configure Claude global MCP settings
2. Restart Claude Code session after MCP configuration
3. Verify MCP tool availability in Claude Code

### Alternative Solution:
If MCP integration continues to fail, the project can proceed with:
- Standard Claude Code tools (Read, Write, Edit, Bash, etc.)
- Direct package commands via Bash tool
- Manual coordination instead of automated swarm features