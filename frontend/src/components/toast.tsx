import { FiAlertCircle, FiCheckCircle, FiX } from "react-icons/fi";

export type ToastType = "success" | "error";

type ToastProps = {
  message: string;
  onDismiss: () => void;
  type: ToastType;
};

/**
 * Displays a top-centered dismissible status toast.
 *
 * @param props - Toast type, message, and dismissal callback.
 * @returns The reusable toast notification component.
 */
export default function Toast({
  message,
  onDismiss,
  type,
}: ToastProps) {
  // Selects accessible styling and semantics based on the toast outcome.
  const isSuccess = type === "success";

  return (
    <div
      aria-live="polite"
      className={`fixed left-1/2 top-5 z-[70] flex w-[calc(100%-2rem)] max-w-xl -translate-x-1/2 items-start gap-3 rounded border px-4 py-3 text-sm shadow-lg ${
        isSuccess
          ? "border-[var(--toast-success-border)] bg-[var(--toast-success-surface)] text-[var(--toast-success-text)]"
          : "border-[var(--toast-error-border)] bg-[var(--toast-error-surface)] text-[var(--toast-error-text)]"
      }`}
      role={isSuccess ? "status" : "alert"}
    >
      {/* Leading icon makes success and error states scannable. */}
      {isSuccess ? (
        <FiCheckCircle aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
      ) : (
        <FiAlertCircle aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
      )}

      <span className="min-w-0 flex-1 font-medium">{message}</span>

      {/* Dismiss control lets users clear the toast before the auto-timeout. */}
      <button
        aria-label="Dismiss notification"
        className="rounded-sm p-0.5 hover:bg-black/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-current"
        onClick={onDismiss}
        type="button"
      >
        <FiX aria-hidden="true" className="size-4" />
      </button>
    </div>
  );
}
