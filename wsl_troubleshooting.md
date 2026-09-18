---
title: WSL Troubleshooting
permalink: /wsl-troubleshooting/
notion:
  title_line: "# WSL Troubleshooting"
  role: troubleshooting
  status: mapped
  page_id: "3dfd9fdd-1a1a-8052-8704-edbfc758ec80"
  url: "https://app.notion.com/p/3dfd9fdd1a1a80528704edbfc758ec80"
---

# WSL Troubleshooting

# No Internet in WSL

**Problem:** UCSF installs Symantec Endpoint protection which has a firewall that is incompatible with WSL

**Solution:** Change the firewall protection to use Windows Defender instead

Step-by-step from my disabling Symantec Endpoint’s firewall:

1. Open Add/Remove Programs
2. On Symantec Endpoint click “…” → Modify
3. Click Next → Modify and you should get to “Custom Setup”
4. Click the (tiny) hard drive icon next to “Network and Host Exploit Mitigation” and select “Entire Feature will be unavailable”
5. “Next” multiple times to complete the install process (I opted out of helping improve their software)
6. You may have to confirm that the installer should be allowed to continue
7. Restart
8. Once it has completed, open Start Menu → “Windows Defender Firewall” (not the one “with Advanced Security”) and confirm that the firewall is active

## Screenshots


![Windows Installed apps: Modify Symantec Endpoint Protection](media/wsl-troubleshooting/step-1.png)

![Symantec Endpoint Protection installation wizard](media/wsl-troubleshooting/step-2.png)

![Program Maintenance: select Modify](media/wsl-troubleshooting/step-3.png)

![Custom Setup: make Network and Host Exploit Mitigation unavailable](media/wsl-troubleshooting/step-4.png)

![Protection Options: LiveUpdate setting](media/wsl-troubleshooting/step-5.png)
