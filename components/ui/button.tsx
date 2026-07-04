import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex min-h-11 items-center justify-center gap-2 rounded-md text-sm font-semibold transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 focus-visible:ring-offset-2 focus-visible:ring-offset-[#070b10] disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        primary: "bg-cyan-500 text-slate-950 shadow-[0_8px_30px_rgba(34,211,238,.18)] hover:bg-cyan-400",
        secondary: "border border-white/10 bg-white/[0.035] text-slate-200 hover:border-white/20 hover:bg-white/[0.07]",
        ghost: "text-slate-400 hover:bg-white/[0.04] hover:text-white"
      },
      size: {
        default: "px-5",
        icon: "h-11 w-11 p-0"
      }
    },
    defaultVariants: { variant: "secondary", size: "default" }
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button ref={ref} className={cn(buttonVariants({ variant, size }), className)} {...props} />
  )
);
Button.displayName = "Button";
