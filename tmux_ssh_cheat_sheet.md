---
title: tmux and SSH Practice
permalink: /tmux-ssh-cheat-sheet/
notion:
  title_line: "# `tmux` and SSH Practice"
  role: reference
  status: mapped
  page_id: "41cd9fdd-1a1a-81d7-ad95-d43f6aeb502c"
  url: "https://app.notion.com/p/41cd9fdd1a1a81d7ad95d43f6aeb502c"
---

# `tmux` and SSH Practice

# Log In with SSH

Master's students should have a login for the course server, `deb-studio.ucsf.edu`. The username and password are in the email from Beverly Bitagon.

**SSH** (secure shell) opens a terminal on another computer. Run these in VS Code's integrated terminal on macOS or in WSL, replacing `USERNAME` with the username from the email.

```bash
ssh USERNAME@deb-studio.ucsf.edu
```

- The first connection asks `Are you sure you want to continue connecting (yes/no/[fingerprint])?` Type `yes`.
- The password prompt shows nothing as you type. Type the password and press Enter.
- The prompt changes to the server's name, such as `USERNAME@deb-studio:~$`. Commands now run on the server.
- `exit` closes the connection and returns to your own computer.

## Reference Card: Connect and Copy

| Command | Purpose | Result |
| :--- | :--- | :--- |
| `ssh USERNAME@deb-studio.ucsf.edu` | Open a shell on the server; `exit` returns | Server prompt |
| `ssh USERNAME@deb-studio.ucsf.edu 'hostname'` | Run one command on the server | The command's output in your terminal |
| `scp results.csv USERNAME@deb-studio.ucsf.edu:~/` | Copy a file from your computer to your server home folder | The file on the server |
| `scp USERNAME@deb-studio.ucsf.edu:~/results.csv .` | Copy a file from the server to the current folder | The file on your computer |
| `scp -r USERNAME@deb-studio.ucsf.edu:~/project .` | Copy a whole folder | The folder on your computer |

Run `scp` from your own computer, not from inside the SSH session.

# Keep Work Running with tmux

**tmux** keeps a terminal session alive on the server after you disconnect, so a long job survives a closed laptop or a dropped connection. Start it after logging in with `ssh`.

Every shortcut starts with the **prefix**, `Ctrl+b`: press it, release, then press the next key.

## Reference Card: tmux Commands

| Task | Command or keys | Result |
| :--- | :--- | :--- |
| Start | `tmux new -s analysis` | New session named `analysis`, status bar at the bottom |
| Detach | `Ctrl+b`, then `d` | Back at the normal prompt; the session keeps running |
| List | `tmux ls` | `analysis: 1 windows (created ...)`, or `no server running` when none exist |
| Reattach | `tmux attach -t analysis` | The session, exactly as you left it |
| New window | `Ctrl+b`, then `c` | A second shell in the same session |
| Switch window | `Ctrl+b`, then `n` (next), `p` (previous), or a window number | The other window |
| Split | `Ctrl+b`, then `%` (side by side) or `"` (stacked) | Two shells on one screen |
| Move between panes | `Ctrl+b`, then an arrow key | The cursor in the other pane |
| Zoom a pane | `Ctrl+b`, then `z`; repeat to undo | One pane fills the screen |
| Scroll | `Ctrl+b`, then `[`; arrow keys or Page Up; `q` to leave | Output that scrolled off the screen |
| Rename session | `Ctrl+b`, then `$` | A new session name |
| List shortcuts | `Ctrl+b`, then `?`; `q` to leave | Every key binding |
| End | `exit` inside the session, or `tmux kill-session -t analysis` from outside | Session closed, stopping anything still running in it |

## Code Snippet: A Job That Survives a Disconnect

On the server, in a project folder with a `uv` environment:

```bash
tmux new -s analysis
source .venv/bin/activate
time python3 analysis.py
```

Press `Ctrl+b`, then `d`, to detach. Now it is safe to type `exit` or close the laptop. Later, log in again and reattach; when the job ends, `time` reports how long it ran:

```bash
ssh USERNAME@deb-studio.ucsf.edu
tmux ls
tmux attach -t analysis
```

# Optional: Log In with an SSH Key

An **SSH key** is a pair of files: a private key that stays on your computer and a public key you copy to the server. With the pair in place, `ssh` and `scp` stop asking for the server password. The steps are the same on macOS and in WSL.

1. Create the key pair on your own computer. Press Enter to accept the default location, then choose a passphrase (or press Enter twice for none).

    ```bash
    ssh-keygen -t ed25519
    ```

    Expect two new files: `~/.ssh/id_ed25519` (private, never share it) and `~/.ssh/id_ed25519.pub` (public).

2. Copy the public key to the server. This asks for the server password one last time.

    ```bash
    ssh-copy-id USERNAME@deb-studio.ucsf.edu
    ```

    Expect `Number of key(s) added: 1`.

3. Log in again. No server password prompt appears.

    ```bash
    ssh USERNAME@deb-studio.ucsf.edu
    ```

4. If you chose a passphrase, unlock the key once per login session so later commands do not ask for it.

    ```bash
    ssh-add
    ```

## Optional: A Short Name for the Server

Add these lines to `~/.ssh/config` on your own computer (create the file if it does not exist):

```text
Host deb-studio
    HostName deb-studio.ucsf.edu
    User USERNAME
```

Then `ssh deb-studio` and `scp results.csv deb-studio:~/` reach the server by that name.

<callout icon="⚠️" color="yellow_bg">
	## The private key stays on your computer
	Never copy `~/.ssh/id_ed25519` to the server, email it, or commit it to Git. Only the `.pub` file leaves your computer.
</callout>
