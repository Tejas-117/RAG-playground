import { type NumericSettingOption } from "@/validation/pipeline-options";

/** Properties used by bounded numeric configuration controls. */
type NumberControlProps = {
  id: string;
  label: string;
  value: string;
  setting: NumericSettingOption;
  onChange: (value: string) => void;
  step?: number;
  helperText?: string;
};

/**
 * Renders a numeric input constrained by backend-provided limits.
 *
 * @param props - Label, value, numeric setting, step, helper text, and change callback.
 * @returns An accessible bounded number control.
 */
export default function NumberControl({
  id,
  label,
  value,
  setting,
  onChange,
  step = 1,
  helperText,
}: NumberControlProps) {
  return (
    /* The field groups its visible label, bounded input, and optional guidance. */
    <label className="block" htmlFor={id}>
      <span
        className={`
          block font-mono text-[10px] font-bold uppercase tracking-[0.12em]
          text-[var(--muted-text)]
        `}
      >
        {label}
      </span>

      <input
        className={`
          mt-2 w-full rounded border border-[var(--border-subtle)]
          bg-[var(--page-surface)] px-3 py-2.5 text-sm font-medium
          text-[var(--charcoal)] outline-none transition-colors
          focus:border-[var(--charcoal)] focus:ring-1 focus:ring-[var(--charcoal)]
        `}
        id={id}
        max={setting.maximum ?? undefined}
        min={setting.minimum}
        onChange={(event) => onChange(event.target.value)}
        step={step}
        type="number"
        value={value}
      />

      {/* Helper text explains constraints not represented by native numeric bounds. */}
      {helperText ? (
        <span className="mt-2 block text-xs leading-5 text-[var(--muted-text)]">{helperText}</span>
      ) : null}
    </label>
  );
}
