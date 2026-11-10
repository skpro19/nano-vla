/**
 * Theme Command Extension
 *
 * Adds a /theme command that switches the interface theme for the current
 * session only. It never writes settings, so other running pi instances and
 * future sessions keep the theme saved in .pi/settings.json.
 *
 * Usage:
 *   /theme              Pick a theme from the list
 *   /theme <name>       Switch to a theme by name (for example yellow-dark)
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

let knownThemeNames: string[] = [];

function themeNames(ctx: ExtensionContext): string[] {
	return ctx.ui.getAllThemes().map((theme) => theme.name);
}

function refreshKnownNames(ctx: ExtensionContext): void {
	knownThemeNames = [...new Set(themeNames(ctx))].sort();
}

function applyTheme(ctx: ExtensionContext, name: string): boolean {
	const theme = ctx.ui.getTheme(name);
	if (theme === undefined) return false;
	ctx.ui.setTheme(theme);
	return true;
}

export default function themeExtension(pi: ExtensionAPI): void {
	pi.on("session_start", (_event, ctx) => {
		refreshKnownNames(ctx);
	});

	pi.registerCommand("theme", {
		description: "Switch the interface theme for this session",
		getArgumentCompletions: (prefix) => {
			const matches = knownThemeNames.filter((name) => name.startsWith(prefix));
			return matches.length > 0 ? matches.map((name) => ({ value: name, label: name })) : null;
		},
		handler: async (args, ctx) => {
			refreshKnownNames(ctx);
			const requested = args.trim();
			if (requested) {
				if (applyTheme(ctx, requested)) {
					ctx.ui.notify(`Theme: ${requested} (this session only)`, "info");
				} else {
					ctx.ui.notify(`Unknown theme "${requested}". Run /theme to pick one.`, "warning");
				}
				return;
			}
			if (!ctx.hasUI) {
				ctx.ui.notify("/theme needs an interactive session", "warning");
				return;
			}
			const names = themeNames(ctx);
			if (names.length === 0) {
				ctx.ui.notify("No themes available", "warning");
				return;
			}
			const selected = await ctx.ui.select("Theme (this session only)", names);
			if (selected === undefined) return;
			if (applyTheme(ctx, selected)) {
				ctx.ui.notify(`Theme: ${selected} (this session only)`, "info");
			}
		},
	});
}
