import { type EvaluationMetricOption } from "@/validation/pipeline-options";

/** Properties for a catalog-backed group of independently selectable metrics. */
type MetricCheckboxGroupProps = {
  id: string;
  label: string;
  options: EvaluationMetricOption[];
  selectedValues: string[];
  onChange: (values: string[]) => void;
  helperText?: string;
};

/**
 * Renders a compact checkbox matrix for selecting any number of evaluation metrics.
 *
 * @param props - Group identity, catalog options, selections, guidance, and change callback.
 * @returns An accessible fieldset that preserves the backend catalog order.
 */
export default function MetricCheckboxGroup({
  id,
  label,
  options,
  selectedValues,
  onChange,
  helperText,
}: MetricCheckboxGroupProps) {
  /**
   * Adds or removes one metric while preserving the backend catalog order.
   *
   * @param metricValue - Stable identifier of the metric being toggled.
   * @param isSelected - Whether the checkbox should be selected after the change.
   * @returns Nothing. The parent receives the complete ordered selection.
   */
  function toggleMetric(metricValue: string, isSelected: boolean): void {
    const nextSelection = options
      .filter((option) =>
        option.value === metricValue ? isSelected : selectedValues.includes(option.value),
      )
      .map((option) => option.value);

    onChange(nextSelection);
  }

  return (
    /* The fieldset presents related metrics as one optional evaluation control. */
    <fieldset aria-describedby={helperText ? `${id}-help` : undefined}>
      <legend
        className={`
          font-mono text-[10px] font-bold uppercase tracking-[0.12em]
          text-[var(--muted-text)]
        `}
      >
        {label}
      </legend>

      {/* Each row keeps the metric explanation attached to its native checkbox. */}
      <div className="mt-2 grid gap-2">
        {options.map((option) => {
          const isSelected = selectedValues.includes(option.value);

          return (
            <label
              className={`
                flex cursor-pointer items-start gap-3 rounded border px-3 py-3
                transition-colors focus-within:ring-2 focus-within:ring-[var(--charcoal)]
                focus-within:ring-offset-1
                ${
                  isSelected
                    ? `border-[var(--charcoal)] bg-[var(--panel-surface)]`
                    : `border-[var(--border-subtle)] bg-[var(--page-surface)]
                    hover:border-[var(--border-strong)]`
                }
              `}
              key={option.value}
            >
              <input
                checked={isSelected}
                className="mt-0.5 size-4 shrink-0 accent-[var(--charcoal)]"
                id={`${id}-${option.value}`}
                onChange={(event) => toggleMetric(option.value, event.target.checked)}
                type="checkbox"
                value={option.value}
              />
              <span className="min-w-0">
                <span className="block text-sm font-semibold text-[var(--charcoal)]">
                  {option.label}
                </span>
                {option.description ? (
                  <span className="mt-1 block text-xs leading-5 text-[var(--muted-text)]">
                    {option.description}
                  </span>
                ) : null}
                {option.requires_reference_answer ? (
                  <span
                    className={`
                      mt-2 inline-flex rounded-sm border border-[var(--border-strong)]
                      px-1.5 py-0.5 font-mono text-[9px] font-bold uppercase
                      tracking-[0.08em] text-[var(--muted-text)]
                    `}
                  >
                    Reference answer required
                  </span>
                ) : null}
              </span>
            </label>
          );
        })}
      </div>

      {/* Guidance makes the empty selection's skip behavior explicit. */}
      {helperText ? (
        <p className="mt-2 text-xs leading-5 text-[var(--muted-text)]" id={`${id}-help`}>
          {helperText}
        </p>
      ) : null}
    </fieldset>
  );
}
