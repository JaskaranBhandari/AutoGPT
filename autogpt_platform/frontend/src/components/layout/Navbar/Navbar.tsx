"use client";

import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { PreviewBanner } from "@/components/layout/Navbar/components/PreviewBanner/PreviewBanner";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { environment } from "@/services/environment";
import { List } from "@phosphor-icons/react";
import { AccountMenu } from "./components/AccountMenu/AccountMenu";
import { LoginButton } from "./components/LoginButton";
import { Wallet } from "./components/Wallet/Wallet";
import { getAccountMenuItems } from "./helpers";

export function Navbar() {
  const { user, isLoggedIn, isUserLoading } = useSupabase();
  const breakpoint = useBreakpoint();
  const isSmallScreen = breakpoint === "sm" || breakpoint === "base";
  const dynamicMenuItems = getAccountMenuItems(user?.role);
  const previewBranchName = environment.getPreviewStealingDev();
  const logoutInProgress = isLogoutInProgress();

  const { data: profile, isLoading: isProfileLoading } = useGetV2GetUserProfile(
    {
      query: {
        select: okData,
        enabled: isLoggedIn && !!user && !logoutInProgress,
        queryKey: ["/api/store/profile", user?.id],
      },
    },
  );

  const isLoadingProfile = isProfileLoading || isUserLoading;
  const shouldShowPreviewBanner = Boolean(isLoggedIn && previewBranchName);

  if (isUserLoading) {
    return null;
  }

  return (
    <>
      {shouldShowPreviewBanner && previewBranchName ? (
        <div className="sticky top-0 z-40 w-full">
          <PreviewBanner branchName={previewBranchName} />
        </div>
      ) : null}

      {!isLoggedIn ? (
        <div className="flex w-full justify-end p-3">
          <LoginButton />
        </div>
      ) : null}

      {/* Desktop top-right: profile + feedback */}
      {isLoggedIn && !isSmallScreen ? (
        <div className="flex items-center justify-end gap-3 px-4 py-3">
          {profile && <Wallet key={profile.username} />}
          <AccountMenu
            userName={profile?.username}
            userEmail={profile?.name}
            avatarSrc={profile?.avatar_url ?? ""}
            menuItemGroups={dynamicMenuItems}
            isLoading={isLoadingProfile}
          />
        </div>
      ) : null}

      {/* Mobile top bar: credits + hamburger on right */}
      {isLoggedIn && isSmallScreen ? (
        <div className="fixed right-0 top-0 z-50 flex items-center gap-2 px-3 py-2">
          <SidebarTrigger className="flex size-10 items-center justify-center rounded-full border border-zinc-200 bg-white [&>svg]:!size-5">
            <List className="!size-5" weight="bold" />
          </SidebarTrigger>
          <Wallet />
        </div>
      ) : null}
    </>
  );
}
