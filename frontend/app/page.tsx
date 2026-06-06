"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { AppShell } from "@/components/layout/app-shell";
import { MobileHeader } from "@/components/layout/mobile-header";
import { Sidebar } from "@/components/sidebar/sidebar";
import { ChatArea } from "@/components/chat/chat-area";
import { SettingsPanel } from "@/components/settings/settings-panel";
import { VoiceModal } from "@/components/voice/voice-modal";
import { CallMeModal } from "@/components/call-me-modal";
import { BackendStatusBanner } from "@/components/common/backend-status-banner";
import { useConversations } from "@/hooks/use-conversations";
import { useHealth } from "@/hooks/use-health";
import { useChat } from "@/hooks/use-chat";
import { useVoice } from "@/hooks/use-voice";
import { storage } from "@/lib/storage";
import type { ChatMessage } from "@/types";

export default function HomePage() {
  // Conversations (client-side persistence).
  const { conversations, activeId, active, newChat, selectChat, deleteChat, setMessages } =
    useConversations();

  // Backend health.
  const { status, health, error, refresh } = useHealth();

  // Per-conversation backend session id (maps to the brain's conversation_id).
  const [sessionId, setSessionId] = useState<string | null>(null);
  useEffect(() => {
    // Reset the backend session when switching conversations.
    setSessionId(null);
  }, [activeId]);

  // Chat.
  const { send, stop, isSending } = useChat({
    sessionId,
    setSessionId,
    messages: active?.messages ?? [],
    setMessages,
  });

  // Voice.
  const appendTurn = useCallback(
    (userMsg: ChatMessage, assistantMsg: ChatMessage) => {
      setMessages((prev) => [...prev, userMsg, assistantMsg]);
    },
    [setMessages],
  );
  const voice = useVoice({ sessionId, setSessionId, onResult: appendTurn });

  // UI state.
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [voiceOpen, setVoiceOpen] = useState(false);
  const [callMeOpen, setCallMeOpen] = useState(false);

  const offline = status === "offline";

  const handleSend = useCallback(
    (text: string) => {
      if (offline) {
        toast.error("Backend offline", { description: "Can't send while the server is unreachable." });
        return;
      }
      void send(text);
    },
    [offline, send],
  );

  const handleNewChat = useCallback(() => {
    newChat();
    setMobileSidebarOpen(false);
  }, [newChat]);

  const handleSelect = useCallback(
    (id: string) => {
      selectChat(id);
      setMobileSidebarOpen(false);
    },
    [selectChat],
  );

  const openVoice = useCallback(() => {
    setVoiceOpen(true);
    void voice.start();
  }, [voice]);

  const handleClearHistory = useCallback(() => {
    storage.clear();
    if (typeof window !== "undefined") window.location.reload();
  }, []);

  const sidebar = (
    <Sidebar
      conversations={conversations}
      activeId={activeId}
      status={status}
      onNewChat={handleNewChat}
      onSelect={handleSelect}
      onDelete={deleteChat}
      onOpenSettings={() => setSettingsOpen(true)}
      onOpenCallMe={() => setCallMeOpen(true)}
    />
  );

  return (
    <>
      <AppShell
        sidebar={sidebar}
        mobileSidebarOpen={mobileSidebarOpen}
        onMobileSidebarOpenChange={setMobileSidebarOpen}
      >
        <MobileHeader
          status={status}
          onOpenSidebar={() => setMobileSidebarOpen(true)}
          onOpenSettings={() => setSettingsOpen(true)}
          onOpenCallMe={() => setCallMeOpen(true)}
        />
        <BackendStatusBanner status={status} onRetry={refresh} />
        <ChatArea
          messages={active?.messages ?? []}
          isSending={isSending}
          disabled={offline}
          voiceActive={voiceOpen && voice.status === "recording"}
          onSend={handleSend}
          onStop={stop}
          onVoice={openVoice}
          onCallMe={() => setCallMeOpen(true)}
        />
      </AppShell>

      <SettingsPanel
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        status={status}
        health={health}
        error={error}
        onRefresh={refresh}
        onClearHistory={handleClearHistory}
      />

      <VoiceModal
        open={voiceOpen}
        onOpenChange={setVoiceOpen}
        status={voice.status}
        error={voice.error}
        last={voice.last}
        audioUrl={voice.audioUrl}
        isSupported={voice.isSupported}
        onStart={voice.start}
        onStop={voice.stop}
        onReset={voice.reset}
      />

      <CallMeModal open={callMeOpen} onOpenChange={setCallMeOpen} />
    </>
  );
}
