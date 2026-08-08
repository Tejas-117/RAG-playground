"use client";

import { type ReactNode, useEffect, useRef } from "react";

type WorkbenchGridCanvasProps = {
  children: ReactNode;
  className?: string;
};

/**
 * Wraps a workbench page in the shared cursor-reactive grid surface.
 *
 * @param props - Child content and optional layout classes for the canvas.
 * @returns A grid-backed section that reacts to fine-pointer movement.
 */
export default function WorkbenchGridCanvas({
  children,
  className = "",
}: WorkbenchGridCanvasProps) {
  // Keeps the grid canvas available to the pointer-tracking effect without causing renders.
  const gridBackgroundRef = useRef<HTMLElement>(null);

  // Subscribes to pointer movement so the shared workbench grid glow follows the cursor.
  useEffect(() => {
    const gridBackground = gridBackgroundRef.current;
    const hasFinePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

    if (gridBackground === null || !hasFinePointer) {
      return;
    }

    // Capturing the narrowed element in a dedicated constant keeps it non-null in callbacks.
    const gridElement: HTMLElement = gridBackground;
    let animationFrameId: number | null = null;
    let nextPointerPosition: { x: number; y: number } | null = null;

    /**
     * Applies the latest pointer position to the grid glow on the next animation frame.
     *
     * @returns Nothing. The grid CSS variables are updated directly on the canvas.
     */
    function paintGridGlow(): void {
      animationFrameId = null;

      if (!nextPointerPosition) {
        return;
      }

      gridElement.style.setProperty("--grid-pointer-x", `${nextPointerPosition.x}px`);
      gridElement.style.setProperty("--grid-pointer-y", `${nextPointerPosition.y}px`);
      nextPointerPosition = null;
    }

    /**
     * Tracks the cursor relative to the grid canvas for the local highlight.
     *
     * @param event - The pointer movement reported by the browser.
     * @returns Nothing. The latest pointer coordinates are queued for painting.
     */
    function handleGridPointerMove(event: PointerEvent): void {
      const bounds = gridElement.getBoundingClientRect();
      nextPointerPosition = {
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      };
      gridElement.style.setProperty("--grid-glow-opacity", "1");

      if (animationFrameId === null) {
        animationFrameId = window.requestAnimationFrame(paintGridGlow);
      }
    }

    /**
     * Fades the grid highlight after the pointer leaves the canvas.
     *
     * @returns Nothing. The glow opacity is reset through a CSS variable.
     */
    function handleGridPointerLeave(): void {
      gridElement.style.setProperty("--grid-glow-opacity", "0");
    }

    gridElement.addEventListener("pointermove", handleGridPointerMove);
    gridElement.addEventListener("pointerleave", handleGridPointerLeave);

    return () => {
      gridElement.removeEventListener("pointermove", handleGridPointerMove);
      gridElement.removeEventListener("pointerleave", handleGridPointerLeave);

      if (animationFrameId !== null) {
        window.cancelAnimationFrame(animationFrameId);
      }
    };
  }, []);

  return (
    /* The shared canvas establishes the grid effect for every non-landing workbench page. */
    <section className={`workbench-grid-background min-w-0 ${className}`} ref={gridBackgroundRef}>
      {children}
    </section>
  );
}
