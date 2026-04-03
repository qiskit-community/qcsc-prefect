from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SubmitError(RuntimeError):
    """Raised when job submission fails."""


class WaitTimeout(RuntimeError):
    """Raised when waiting for final job status times out."""


class CancelError(RuntimeError):
    """Raised when job cancellation fails."""


async def run_command(*args: str, cwd: Path | None = None) -> str:
    """Run a command asynchronously and return decoded stdout."""

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_b, err_b = await proc.communicate()
    out = (out_b or b"").decode(errors="replace")
    err = (err_b or b"").decode(errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)} rc={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}")
    return out


@dataclass(frozen=True)
class SubmitResult:
    """Submission result payload returned by runtime ``submit`` methods."""

    job_id: str
    raw_output: str


def _is_terminal_state(state: str) -> bool:
    normalized = state.strip().split()[0].rstrip("+")
    return normalized in {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }


class SlurmRuntime:
    """Minimal async runtime for Slurm ``sbatch``/``sacct``/``scancel`` commands."""

    async def submit(self, script_path: Path, *, cwd: Path | None = None) -> SubmitResult:
        """Submit a Slurm script with ``sbatch --parsable``."""

        try:
            stdout = await run_command("sbatch", "--parsable", str(script_path), cwd=cwd)
        except Exception as e:
            raise SubmitError(f"sbatch failed for {script_path}") from e

        out = stdout.strip()
        if not out:
            raise SubmitError("sbatch returned empty stdout; cannot parse job id.")
        job_id = out.split(";", 1)[0].strip()
        if not job_id:
            raise SubmitError(f"Failed to parse Slurm job id from sbatch output: {out}")
        return SubmitResult(job_id=job_id, raw_output=out)

    async def wait_final_status(
        self,
        job_id: str,
        *,
        watch_poll_interval: float = 10.0,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Poll ``sacct`` until the job reaches a terminal state."""

        start = asyncio.get_running_loop().time()
        try:
            while True:
                if timeout_seconds is not None:
                    now = asyncio.get_running_loop().time()
                    if now - start > timeout_seconds:
                        raise WaitTimeout(f"timeout waiting for job_id={job_id}")

                stdout = await run_command(
                    "sacct",
                    "-j",
                    job_id,
                    "--format=JobID,State,ExitCode,Elapsed,AllocCPUS,NodeList",
                    "--parsable2",
                    "--noheader",
                )

                for line in stdout.splitlines():
                    fields = line.split("|")
                    if len(fields) < 6:
                        continue
                    job_id_field = fields[0].strip()
                    if not job_id_field or job_id_field != job_id:
                        continue
                    state = fields[1].strip()
                    out = {
                        "JobID": job_id_field,
                        "State": state,
                        "ExitCode": fields[2].strip(),
                        "Elapsed": fields[3].strip(),
                        "AllocCPUS": fields[4].strip(),
                        "NodeList": fields[5].strip(),
                    }
                    if _is_terminal_state(state):
                        return out

                await asyncio.sleep(watch_poll_interval)

        except asyncio.CancelledError:
            await run_command("scancel", job_id)
            return {}

    async def cancel(self, job_id: str) -> None:
        """Cancel a Slurm job using ``scancel``."""

        try:
            await run_command("scancel", job_id)
        except Exception as e:
            raise CancelError(f"scancel failed for job_id={job_id}") from e
