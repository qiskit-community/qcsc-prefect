# Workflow for observability demo on Miyabi
#
# Author: Naoki Kanazawa (knzwnao@jp.ibm.com)

import io
import json
import os
import uuid
from typing import Any, cast
from uuid import UUID

import numpy as np
from prefect.artifacts import TableArtifact
from prefect.client.orchestration import get_client
from prefect.client.schemas.actions import ArtifactUpdate
from prefect.client.schemas.filters import (
    ArtifactFilter,
    ArtifactFilterKey,
    FlowRunFilter,
    FlowRunFilterId,
)
from prefect.context import FlowRunContext
from prefect_aws.s3 import S3Bucket


def save_ndarray(
    file_prefix: str,
    **arrays: np.ndarray,
) -> str | None:
    s3_client = S3Bucket.load("s3-sqd")
    ctx_flow = FlowRunContext.get()
    s3_client.bucket_folder = os.path.join(
        ctx_flow.flow.name,
        ctx_flow.flow_run.name,
    )

    with io.BytesIO() as buf:
        np.savez_compressed(buf, **arrays, allow_pickle=False)
        buf.seek(0)
        filepath = s3_client.upload_from_file_object(
            buf,
            to_path=f"{file_prefix}_{uuid.uuid4().hex}.npz",
        )
    return filepath


def load_ndarray(
    file_path: str,
    key: str,
) -> np.ndarray:
    s3_client = S3Bucket.load("s3-sqd")

    with io.BytesIO() as buf:
        s3_client.download_object_to_file_object(file_path, buf)
        buf.seek(0)
        array = np.load(buf).get(key)
    return array


def extend_table_artifact(
    artifact_key: str,
    new_table: list[dict[str, Any]] | list[list[Any]],
    index: int = 0,
) -> UUID:
    ctx_flow = FlowRunContext.get()
    
    with get_client(sync_client=True) as client:
        artifacts = client.read_artifacts(
            artifact_filter=ArtifactFilter(
                key=ArtifactFilterKey(any_=[artifact_key]),
            ),
            flow_run_filter=FlowRunFilter(
                id=FlowRunFilterId(any_=[ctx_flow.flow_run.id]),
            ),
        )
        artifact = artifacts[index]
        current_data = json.loads(artifact.data)
        
        assert isinstance(current_data, list)
        current_data.extend(new_table)

        update = ArtifactUpdate(
            data=cast(str, TableArtifact(table=current_data).format())
        )
        client.update_artifact(
            artifact_id=artifact.id,
            artifact=update,
        )    
    return artifact.id
