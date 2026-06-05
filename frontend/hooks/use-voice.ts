"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { base64ToAudioUrl, uid } from "@/lib/utils";
import { ApiError, type ChatMessage, type VoiceResponse } from "@/types";

export type VoiceStatus = "idle" | "requesting" | "recording" | "processing" | "error";

interface UseVoiceArgs {
  sessionId: string | null;
  setSessionId: (id: string) => void;
  /** Append the recognised turn (user transcript + assistant answer) to chat. */
  onResult?: (userMsg: ChatMessage, assistantMsg: ChatMessage) => void;
}

export function useVoice({ sessionId, setSessionId, onResult }: UseVoiceArgs) {
  const [status, setStatus] = useState<VoiceStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [last, setLast] = useState<VoiceResponse | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const isSupported =
    typeof window !== "undefined" &&
    typeof navigator !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof window.MediaRecorder !== "undefined";

  const cleanupStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  useEffect(() => {
    return () => {
      cleanupStream();
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const process = useCallback(
    async (blob: Blob) => {
      setStatus("processing");
      try {
        const res = await api.voiceAudio(blob, { sessionId, speak: true });
        if (res.session_id) setSessionId(res.session_id);
        setLast(res);

        const now = Date.now();
        const userMsg: ChatMessage = {
          id: uid("msg"),
          role: "user",
          content: res.transcript?.trim() || "🎤 (voice message)",
          createdAt: now,
          status: "complete",
        };
        const assistantMsg: ChatMessage = {
          id: uid("msg"),
          role: "assistant",
          content: res.answer,
          createdAt: now + 1,
          status: "complete",
          citations: res.citations,
          toolCalls: res.tool_calls,
          latencyMs: res.latency_ms,
        };
        onResult?.(userMsg, assistantMsg);

        if (res.audio_base64) {
          const url = base64ToAudioUrl(res.audio_base64);
          setAudioUrl((prev) => {
            if (prev) URL.revokeObjectURL(prev);
            return url;
          });
          const el = audioRef.current ?? new Audio();
          audioRef.current = el;
          el.src = url;
          el.play().catch(() => void 0);
        }
        setStatus("idle");
      } catch (err) {
        const message =
          err instanceof ApiError ? err.message : "Voice request failed. Please try again.";
        setError(message);
        setStatus("error");
        toast.error("Voice error", { description: message });
      }
    },
    [onResult, sessionId, setSessionId],
  );

  const start = useCallback(async () => {
    if (!isSupported) {
      const message = "Your browser doesn't support audio recording.";
      setError(message);
      setStatus("error");
      toast.error("Voice unavailable", { description: message });
      return;
    }
    setError(null);
    setStatus("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mime = MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : undefined;
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        cleanupStream();
        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        if (blob.size === 0) {
          setStatus("idle");
          return;
        }
        void process(blob);
      };
      recorder.start();
      recorderRef.current = recorder;
      setStatus("recording");
    } catch {
      const message = "Microphone access was denied.";
      setError(message);
      setStatus("error");
      cleanupStream();
      toast.error("Microphone blocked", { description: message });
    }
  }, [cleanupStream, isSupported, process]);

  const stop = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setError(null);
    setLast(null);
  }, []);

  return { status, error, last, audioUrl, isSupported, start, stop, reset };
}
