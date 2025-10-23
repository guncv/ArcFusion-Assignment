#!/usr/bin/env python3
"""
PDF Ingestion CLI Tool

A production-ready command-line tool for ingesting PDF documents into the RAG system.

Usage:
    python ingest_documents.py /path/to/document.pdf
    python ingest_documents.py /path/to/documents/*.pdf
    python ingest_documents.py --clear /path/to/document.pdf
    python ingest_documents.py --directory /path/to/docs/

Examples:
    # Ingest a single PDF
    python ingest_documents.py documents/manual.pdf

    # Ingest multiple PDFs
    python ingest_documents.py docs/doc1.pdf docs/doc2.pdf docs/doc3.pdf

    # Ingest all PDFs in a directory
    python ingest_documents.py --directory documents/

    # Clear existing collection and ingest
    python ingest_documents.py --clear documents/manual.pdf

    # Ingest with custom collection name
    python ingest_documents.py --collection my_docs documents/*.pdf
"""

import sys
import argparse
import asyncio
from pathlib import Path
from typing import List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from infrastructure.rag import get_rag_pipeline
from core.log.logger import logger


console = Console()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf
  %(prog)s doc1.pdf doc2.pdf doc3.pdf
  %(prog)s --directory ./documents
  %(prog)s --clear --directory ./documents
        """
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="PDF file paths to ingest"
    )

    parser.add_argument(
        "-d", "--directory",
        type=str,
        help="Directory containing PDF files to ingest"
    )

    parser.add_argument(
        "-c", "--clear",
        action="store_true",
        help="Clear existing collection before ingesting"
    )

    parser.add_argument(
        "--collection",
        type=str,
        help="Custom collection name (uses config default if not specified)"
    )

    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively search directory for PDFs"
    )

    return parser.parse_args()


def gather_pdf_files(args: argparse.Namespace) -> List[Path]:
    """
    Gather PDF files from arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        List of PDF file paths
    """
    pdf_files = []

    # Gather files from explicit paths
    if args.files:
        for file_path in args.files:
            path = Path(file_path)
            if path.is_file() and path.suffix.lower() == ".pdf":
                pdf_files.append(path)
            elif path.is_file():
                console.print(f"[yellow]Warning:[/yellow] Skipping non-PDF file: {file_path}")
            else:
                console.print(f"[red]Error:[/red] File not found: {file_path}")

    # Gather files from directory
    if args.directory:
        dir_path = Path(args.directory)
        if not dir_path.is_dir():
            console.print(f"[red]Error:[/red] Directory not found: {args.directory}")
            sys.exit(1)

        # Use recursive or non-recursive glob
        pattern = "**/*.pdf" if args.recursive else "*.pdf"
        found_pdfs = list(dir_path.glob(pattern))

        if not found_pdfs:
            console.print(
                f"[yellow]Warning:[/yellow] No PDF files found in {args.directory}"
            )
        else:
            pdf_files.extend(found_pdfs)

    return pdf_files


def display_files_table(pdf_files: List[Path]) -> None:
    """Display table of files to be ingested."""
    table = Table(title="PDF Files to Ingest", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("File Name", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Size", justify="right", style="yellow")

    for i, pdf_path in enumerate(pdf_files, 1):
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        table.add_row(
            str(i),
            pdf_path.name,
            str(pdf_path.parent),
            f"{size_mb:.2f} MB"
        )

    console.print(table)


async def ingest_pdfs(
    pdf_files: List[Path],
    clear_existing: bool = False
) -> None:
    """
    Ingest PDF files into the RAG system.

    Args:
        pdf_files: List of PDF file paths
        clear_existing: Whether to clear existing collection
    """
    if not pdf_files:
        console.print("[red]Error:[/red] No PDF files to ingest")
        sys.exit(1)

    # Display files
    console.print()
    display_files_table(pdf_files)
    console.print()

    # Confirm action
    if clear_existing:
        console.print(
            "[bold red]Warning:[/bold red] This will clear the existing collection!"
        )

    # Initialize RAG pipeline
    console.print("[bold blue]Initializing RAG pipeline...[/bold blue]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading RAG pipeline...", total=None)

            progress.update(task, description="Processing documents...")

            # Ingest documents
            pdf_paths_str = [str(p) for p in pdf_files]
            pipeline = get_rag_pipeline()
            result = pipeline.ingest_documents(
                pdf_paths=pdf_paths_str,
                clear_existing=clear_existing
            )

            progress.update(task, description="Ingestion complete!", completed=True)

        # Display results
        console.print()
        if result.get("status") == "success":
            console.print(Panel.fit(
                f"[bold green]✓ Ingestion Successful![/bold green]\n\n"
                f"PDFs Processed: {result.get('pdfs_processed', 0)}\n"
                f"Chunks Created: {result.get('chunks_created', 0)}\n"
                f"Documents Indexed: {result.get('documents_indexed', 0)}\n"
                f"Collection: {result.get('collection_name', 'N/A')}",
                title="Ingestion Results",
                border_style="green"
            ))
        else:
            console.print(Panel.fit(
                f"[bold red]✗ Ingestion Failed[/bold red]\n\n"
                f"Error: {result.get('error', 'Unknown error')}",
                title="Ingestion Results",
                border_style="red"
            ))
            sys.exit(1)

    except Exception as e:
        console.print()
        console.print(Panel.fit(
            f"[bold red]✗ Ingestion Error[/bold red]\n\n"
            f"[red]{type(e).__name__}:[/red] {str(e)}",
            title="Error",
            border_style="red"
        ))
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    # Parse arguments
    args = parse_arguments()

    # Validate input
    if not args.files and not args.directory:
        console.print("[red]Error:[/red] No input files or directory specified")
        console.print("Use --help for usage information")
        sys.exit(1)

    # Gather PDF files
    console.print("[bold]PDF Ingestion Tool[/bold]")
    console.print("=" * 50)

    pdf_files = gather_pdf_files(args)

    if not pdf_files:
        console.print("[red]Error:[/red] No valid PDF files found")
        sys.exit(1)

    # Run ingestion
    asyncio.run(ingest_pdfs(pdf_files, args.clear))


if __name__ == "__main__":
    main()
