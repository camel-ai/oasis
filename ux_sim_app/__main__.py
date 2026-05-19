"""Entry point: python -m ux_sim_app"""
import argparse
from ux_sim_app.ui.app import launch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OASIS UX Simulation App")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()
    launch(port=args.port, share=args.share)
