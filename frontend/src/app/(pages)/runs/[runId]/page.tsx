import RunDetailWorkbench from "@/components/run-detail-workbench";

/** Render the sample inspection workspace for a requested run path. */
export default async function RunDetailPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;

  return <RunDetailWorkbench requestedRunId={runId} />;
}
