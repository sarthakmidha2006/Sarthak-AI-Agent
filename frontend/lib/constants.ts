import type { SourceKind } from "@/types";
import {
  FileText,
  Github,
  FileCode2,
  FolderGit2,
  Briefcase,
  User,
  FileQuestion,
  type LucideIcon,
} from "lucide-react";

/** Persona identity shown in the profile card. */
export const PERSONA = {
  name: "Sarthak Midha",
  roles: ["AI Engineer", "Product Designer"],
  org: "Scaler School of Technology",
  tagline: "Building AI-native products from a verified, grounded corpus.",
  initials: "SM",
  skills: [
    "Python",
    "TypeScript",
    "FastAPI",
    "Next.js",
    "RAG",
    "LangChain",
    "PostgreSQL",
    "Tailwind CSS",
  ],
  interests: [
    "Retrieval-Augmented Generation",
    "Agentic workflows",
    "LLM evaluation",
    "Voice interfaces",
    "Human-AI interaction",
  ],
  stats: [
    { label: "Projects", value: "12+" },
    { label: "Focus", value: "AI / RAG" },
    { label: "Stack", value: "Full-stack" },
  ],
} as const;

/** Quick-question chips. Each sends `query` into the chat. */
export const QUICK_QUESTIONS: { label: string; query: string }[] = [
  { label: "Tell me about Sarthak", query: "Tell me about Sarthak Midha." },
  { label: "What projects has he built?", query: "What projects has Sarthak built?" },
  { label: "What did he do at Cyparta?", query: "What did Sarthak do at Cyparta?" },
  { label: "What are his AI interests?", query: "What are Sarthak's AI interests?" },
  {
    label: "What technologies does he know?",
    query: "What technologies and tools does Sarthak know?",
  },
  { label: "Schedule a meeting", query: "I'd like to schedule a meeting with Sarthak." },
];

/** Visual config for each source badge kind. */
export const SOURCE_BADGES: Record<
  SourceKind,
  { label: string; icon: LucideIcon; className: string }
> = {
  resume: {
    label: "Resume",
    icon: FileText,
    className: "bg-blue-500/12 text-blue-300 border-blue-500/25",
  },
  github: {
    label: "GitHub",
    icon: Github,
    className: "bg-zinc-500/12 text-zinc-200 border-zinc-400/25",
  },
  markdown: {
    label: "Markdown",
    icon: FileCode2,
    className: "bg-emerald-500/12 text-emerald-300 border-emerald-500/25",
  },
  project: {
    label: "Project",
    icon: FolderGit2,
    className: "bg-fuchsia-500/12 text-fuchsia-300 border-fuchsia-500/25",
  },
  experience: {
    label: "Experience",
    icon: Briefcase,
    className: "bg-amber-500/12 text-amber-300 border-amber-500/25",
  },
  about: {
    label: "About",
    icon: User,
    className: "bg-violet-500/12 text-violet-300 border-violet-500/25",
  },
  unknown: {
    label: "Source",
    icon: FileQuestion,
    className: "bg-white/5 text-muted-foreground border-white/10",
  },
};

/** Names of scheduling tools the backend may surface in tool_calls. */
export const AVAILABILITY_TOOL_NAMES = ["check_availability", "get_availability"];
export const BOOKING_TOOL_NAMES = ["book_meeting", "book"];
