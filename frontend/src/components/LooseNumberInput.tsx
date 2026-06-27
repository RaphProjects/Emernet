import { useEffect, useRef, useState } from 'react';
import type React from 'react';

type LooseNumberInputProps = Omit<
  React.InputHTMLAttributes<HTMLInputElement>,
  'type' | 'value' | 'onChange' | 'min' | 'max'
> & {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  fallback?: number;
};

function clamp(value: number, min?: number, max?: number) {
  let next = value;
  if (typeof min === 'number') next = Math.max(min, next);
  if (typeof max === 'number') next = Math.min(max, next);
  return next;
}

function isCompleteNumber(raw: string) {
  const trimmed = raw.trim();
  if (!trimmed || trimmed === '-' || trimmed === '+' || trimmed === '.' || trimmed === '-.' || trimmed === '+.') {
    return false;
  }
  return Number.isFinite(Number(trimmed));
}

export default function LooseNumberInput({
  value,
  onChange,
  min,
  max,
  fallback,
  onBlur,
  onKeyDown,
  ...props
}: LooseNumberInputProps) {
  const ref = useRef<HTMLInputElement>(null);
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    if (document.activeElement !== ref.current) {
      setDraft(String(value));
    }
  }, [value]);

  const commit = () => {
    const parsed = isCompleteNumber(draft) ? Number(draft) : (fallback ?? value ?? min ?? 0);
    const next = clamp(parsed, min, max);
    setDraft(String(next));
    onChange(next);
  };

  return (
    <input
      {...props}
      ref={ref}
      type="number"
      min={min}
      max={max}
      value={draft}
      onChange={event => {
        const raw = event.target.value;
        setDraft(raw);
        if (isCompleteNumber(raw)) {
          onChange(Number(raw));
        }
      }}
      onBlur={event => {
        commit();
        onBlur?.(event);
      }}
      onKeyDown={event => {
        if (event.key === 'Enter') {
          ref.current?.blur();
        }
        onKeyDown?.(event);
      }}
    />
  );
}
