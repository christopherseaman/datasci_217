"""Public artifact checks for Assignment 01."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re

ROSTER_HASHES = frozenset({
    "07f66adfe2a38fb3a84a2ab2f55d40bdb5cabd85355ecb9f65290385c5cc76b8",
    "1021e732596c04d56e2aa68709e701bb7ea4136bd24c536a3c51997882dd43f2",
    "10ece8c62bc81e6caa64b78a1040b29a1604f84761ba6df6e4495bd15a463b2e",
    "18bb6954a841ec03028a44e250628fc3575e911bde64b5fd8611238d66e65a2b",
    "24a520cb993f3b75c6741cab1e0ff53376e96370a64191a559afb99faa80a6fe",
    "289dba23316b3e2b10b677f966b3ec29a5277ced8af256bb54ea3d14b0b5e00e",
    "39ab2539040fa52910bcff9b9ede5dd578adf9a4c21edd7af70fd236de85ff0b",
    "40bc50894e9238057d3f61d96c89c43fde656a0daa4e57cc297b7a4b41ea9822",
    "4231dafea7f15471053e731f9f583447cf1a97fa98edf01ec95e9681321c6270",
    "49cc715f310f0ba0e0fc33a23db7dcc9656ac312c24d32f9424dc8c333a22f29",
    "5cbf859665a8ea51a585d25a929b4d0ecceb619e17e2413d4ad29fa866dcb88e",
    "5eaf3d8fedcbaa1a0230e3886444a36b0c2c8e35fa519862e1def1ca829717fb",
    "62b4a37e8eb0ba7d6f6a8c97cacc3460c4fc0344ed3437ebc5b893e77b854676",
    "62cea3fcba60444e3c3d22fada8814ede08a4576e2294d7668986280719d5384",
    "65744a984caca19adf80e29930bab1d0edfb3009e54f9427ec9747c25073d84f",
    "724c02d046a6f4c13cd76d29a9297cc1fbe27775e0e7ba6680b151e48e634bd5",
    "7ae0d4645e632ac8e54f92bb38adf30b819e82dff1f71476d7b795cc8ae84223",
    "819dd0b63f9c1cb9cc1dc20e11d099e8b7d06c499314000c27600024f5e35220",
    "8c7aacd0f592fbff2ce69a5cfd51bbd136133752c6b11960fe9849e3dfff0aff",
    "919b5177debfaa9db39d1cfa54eb38770be99c312cd76a048dc7e9bdfc344513",
    "94fb8f3cc19b9e6599824fb7a6ed055a2be650221a8289af1d10971fb33ab0c3",
    "97983c85fbb08e26a8e7c05829d614313c94b46bb8d3453b76dee3fb1cb408b8",
    "9cfe029fece611598ed5d758b7a537c9d9d65d182c21a2a06fe82273a5e77b8e",
    "9d6815a23b72f3770f107a5371926d63a01d09e5ce13eb8651d8bc7c76a2ff68",
    "9e40278b71e32198b63bd6e91b33257f6b39cc49fa8368478c65b1c9e70bf086",
    "9fd76c7d1eafa348c5c325d9b1f6c1b91091875444c0b0d6514be57be272fa2b",
    "a128b3dd4426613a4c9ecd0e29b27d2a26bd51d4e713605e36495f02b079e6b1",
    "aacfa06ad5191c6ca7c27bf740e5522ff1f696547c34730e0278459036f7e5bb",
    "abcd16e55a833d6423e31cc2d5821dadb6faa5bcfb0e932f4549d4f8570395b1",
    "ac21180378214fde49624883c961affef8b76e6e3ee17934fb32182a65e4df17",
    "aff17118cad18b223ff2d3e6483d55a0fc0436812463a310e237b4e6d65e6de5",
    "b1968e7d557ab56b575533fe2250a0e85634c70cd7de9b966b4e94097e39b6a2",
    "b21f05d20a6f9a585ab42ffce153a3209943dee30a6f1e3b58c110c719113a54",
    "c0a3423603e25aee7189168c4117b0e619c79f43b95cbf27877dcd1078605f7b",
    "cabb9a5e2202f47f3da834e7a69c59b238cdf438c153d14d4920454a32fc6ef4",
    "d6d383fdcf137c98e15d60afd2312076168e3ddd6428a478c6b46f63e3c276ba",
    "dc0dbc8c08708f765945d5f06bb6fc73f8d92234ff291652ec645c8548e467a5",
    "e6ca7c71e53b865114afa8c65561b40edabb8ff1d6972b6e4a85d49cd3263ef3",
    "f79c0262a92f5def6ae726c8850be7b29b6d63977a8e6dbddf90c8756efd8226",
    "fee59a908616085e5e356ae776eb1955ba84a4bcd97be649c7749be4ea294346",
})


EXPECTED_READINESS = """Python family: 3.13
Project: DataSci 217 Assignment 01
Script: readiness.py
Measurement: 18 within range
Measurement: 21 review
Measurement: 24 review
Measurement: 19 within range
Count: 4
Total: 82
Mean: 20.5
Review count: 2
Readiness: complete
Participant count: 4
Next checkpoint: 5
"""


@dataclass(frozen=True)
class PublicCheck:
    name: str
    action: Callable[[Path], None]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_terminal_practice(root: Path) -> None:
    practice = root / "terminal-practice"
    _assert(practice.is_dir() and not practice.is_symlink(), "Create a regular terminal-practice directory.")
    for name in ("source.txt", "path-check.txt"):
        _assert((practice / name).is_file() and not (practice / name).is_symlink(), f"terminal-practice/{name} must be a regular file.")


def check_output_artifact(root: Path) -> None:
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "Create a regular output/ directory.")
    report = output / "readiness.txt"
    _assert(report.is_file() and not report.is_symlink(), "Commit output/readiness.txt as a regular file.")
    try:
        stored = report.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise AssertionError("output/readiness.txt must be UTF-8 text.") from error
    _assert(stored == EXPECTED_READINESS, "output/readiness.txt must contain the documented 14-line readiness report.")
    identity = output / "student_identity.txt"
    _assert(identity.is_file() and not identity.is_symlink(), "Run capture_identity.py and commit output/student_identity.txt.")
    try:
        identity_text = identity.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise AssertionError("student_identity.txt must be UTF-8 text.") from error
    identity_hash = identity_text.strip().lower()
    _assert(re.fullmatch(r"[0-9a-f]{64}", identity_hash) is not None, "student_identity.txt must contain one SHA-256 hash; surrounding whitespace and letter case are ignored.")
    _assert(identity_hash in ROSTER_HASHES, "The identity hash does not match the course roster. Rerun capture_identity.py with your course roster email; contact the course team if it still fails.")


PUBLIC_CHECKS = (
    PublicCheck("terminal practice evidence", check_terminal_practice),
    PublicCheck("committed readiness and identity artifacts", check_output_artifact),
)


def run_public_checks(root: Path) -> list[tuple[str, str | None]]:
    results = []
    for check in PUBLIC_CHECKS:
        try:
            check.action(root)
        except (AssertionError, OSError) as error:
            results.append((check.name, str(error)))
        else:
            results.append((check.name, None))
    return results
