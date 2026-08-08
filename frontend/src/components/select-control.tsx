import { FiChevronDown } from "react-icons/fi";
import { type PipelineOption } from "@/validation/pipeline-options";

/** Properties shared by catalog-backed select controls. */
type SelectControlProps = {
  id: string;
  label: string;
  value: string;
  options: PipelineOption[];
  onChange: (value: string) => void;
  helperText?: string;
};

/**
 * Renders a select whose choices come directly from the pipeline catalog.
 *
 * @param props - Label, value, catalog options, helper text, and change callback.
 * @returns An accessible catalog-backed select control.
 */
export default function SelectControl({
  id,
  label,
  value,
  options,
  onChange,
  helperText,
}: SelectControlProps) {
  return (
    /* The field groups its visible label, native select, and optional guidance. */
    <label className="block" htmlFor={id}>
      <span
        className={`
          block font-mono text-[10px] font-bold uppercase tracking-[0.12em]
          text-[var(--muted-text)]
        `}
      >
        {label}
      </span>

      {/* The select wrapper anchors the decorative dropdown indicator. */}
      <span className="relative mt-2 block">
        <select
          className={`
            w-full appearance-none rounded border border-[var(--border-subtle)]
            bg-[var(--page-surface)] px-3 py-2.5 pr-9 text-sm font-medium
            text-[var(--charcoal)] outline-none transition-colors
            focus:border-[var(--charcoal)] focus:ring-1 focus:ring-[var(--charcoal)]
          `}
          id={id}
          onChange={(event) => onChange(event.target.value)}
          value={value}
        >
          {/* Every choice uses the stable value supplied by FastAPI. */}
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>

        <FiChevronDown
          aria-hidden="true"
          className={`
            pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2
            text-[var(--muted-text)]
          `}
        />
      </span>

      {/* Helper text explains requirements or behavior for the current selection. */}
      {helperText ? (
        <span className="mt-2 block text-xs leading-5 text-[var(--muted-text)]">{helperText}</span>
      ) : null}
    </label>
  );
}
