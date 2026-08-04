import Link from "next/link";
import {
  FiBarChart2,
  FiFileText,
  FiGrid,
  FiHelpCircle,
  FiPieChart,
} from "react-icons/fi";

const navigationItems = [
  { label: "Documents", icon: FiFileText, active: true },
  { label: "Experiments", icon: FiPieChart, active: false },
  { label: "Run Dashboard", icon: FiGrid, active: false },
  { label: "Evaluation", icon: FiBarChart2, active: false },
];

/**
 * Presents the desktop navigation pane for the RAG experiment workbench.
 *
 * @returns The fixed workbench sidebar with primary and help navigation.
 */
export default function WorkbenchSidebar() {
  return (
    <aside className="hidden min-h-screen flex-col justify-between border-r border-[var(--border-subtle)] bg-[var(--panel-surface)] px-3 py-6 lg:flex">
      {/* Primary navigation identifies the active workbench area. */}
      <nav aria-label="Workbench navigation" className="space-y-1">
        {navigationItems.map((item) => {
          const Icon = item.icon;

          return (
            <Link
              aria-current={item.active ? "page" : undefined}
              className={`flex items-center gap-3 rounded-sm px-3 py-2 text-sm font-medium transition-colors ${
                item.active
                  ? "bg-[var(--border-subtle)] text-[var(--charcoal)]"
                  : "text-[var(--tone-black)] hover:bg-[var(--hover-surface)]"
              }`}
              href={item.active ? "/ingestion" : "#"}
              key={item.label}
            >
              <Icon aria-hidden="true" className="size-5" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Secondary navigation keeps support separate from core workflows. */}
      <a
        className="flex items-center gap-2 px-3 text-xs font-medium text-[var(--tone-black)] hover:text-[var(--charcoal)]"
        href="#"
      >
        <FiHelpCircle aria-hidden="true" className="size-4" />
        Help
      </a>
    </aside>
  );
}
