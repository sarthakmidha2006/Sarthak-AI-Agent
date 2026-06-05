import { Fragment } from "react";

/**
 * Minimal, XSS-safe rich-text renderer for assistant answers. Handles
 * **bold**, `inline code`, [n] citation markers, and paragraph/line breaks.
 * Deliberately avoids dangerouslySetInnerHTML — everything is React nodes.
 */
function renderInline(text: string): React.ReactNode[] {
  // Tokenize on **bold**, `code`, and [n] citation markers.
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[\d+\])/g;
  const parts = text.split(pattern).filter(Boolean);
  return parts.map((part, i) => {
    if (/^\*\*[^*]+\*\*$/.test(part)) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (/^`[^`]+`$/.test(part)) {
      return <code key={i}>{part.slice(1, -1)}</code>;
    }
    if (/^\[\d+\]$/.test(part)) {
      return (
        <sup
          key={i}
          className="mx-0.5 rounded bg-primary/15 px-1 text-[10px] font-semibold text-primary"
        >
          {part.slice(1, -1)}
        </sup>
      );
    }
    return <Fragment key={i}>{part}</Fragment>;
  });
}

export function FormattedText({ text }: { text: string }) {
  const blocks = text.split(/\n{2,}/);
  return (
    <div className="answer-prose text-[15px] text-foreground/90">
      {blocks.map((block, i) => {
        const lines = block.split("\n");
        const isBullet = lines.every((l) => /^\s*[-*]\s+/.test(l));
        const isNumbered = lines.every((l) => /^\s*\d+\.\s+/.test(l));
        if (isBullet) {
          return (
            <ul key={i}>
              {lines.map((l, j) => (
                <li key={j}>{renderInline(l.replace(/^\s*[-*]\s+/, ""))}</li>
              ))}
            </ul>
          );
        }
        if (isNumbered) {
          return (
            <ol key={i}>
              {lines.map((l, j) => (
                <li key={j}>{renderInline(l.replace(/^\s*\d+\.\s+/, ""))}</li>
              ))}
            </ol>
          );
        }
        return (
          <p key={i}>
            {lines.map((l, j) => (
              <Fragment key={j}>
                {renderInline(l)}
                {j < lines.length - 1 && <br />}
              </Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}
