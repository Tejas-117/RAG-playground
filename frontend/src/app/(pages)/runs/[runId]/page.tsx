import RunDetailWorkbench from "@/components/run-detail-workbench";

/** Render API-backed inspection for the requested persisted benchmark run. */
export default async function RunDetailPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;

  return <RunDetailWorkbench requestedRunId={runId} />;
}
