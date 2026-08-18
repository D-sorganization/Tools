/** Bounded browser file boundary for exact regional-ground execution jobs. */

import {
  MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
  regionalGroundExecutionJobFromJson,
  stableRegionalGroundExecutionJobJson,
  type RegionalGroundExecutionJob,
} from "./regionalGroundExecutionJob";
import { readBoundedUtf8File, type BoundedUtf8File } from "./boundedUtf8File";

export type RegionalGroundExecutionJobFile = BoundedUtf8File;

export const readRegionalGroundExecutionJobFile = async (
  file: RegionalGroundExecutionJobFile,
): Promise<RegionalGroundExecutionJob> => {
  const source = await readBoundedUtf8File(
    file,
    MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
    "regional-ground execution job",
  );
  return regionalGroundExecutionJobFromJson(source);
};

/** Start a canonical copy download of the accepted exact job. */
export const downloadRegionalGroundExecutionJob = (
  job: RegionalGroundExecutionJob,
): void => {
  const url = URL.createObjectURL(new Blob(
    [stableRegionalGroundExecutionJobJson(job)],
    { type: "application/json;charset=utf-8" },
  ));
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${job.job_id}.regional-ground-job.json`;
    anchor.click();
  } finally {
    URL.revokeObjectURL(url);
  }
};
