#!/usr/bin/env python
"""
Script voor het uitvoeren van tests en het genereren van een coverage rapport.

Gebruik:
    python run_tests.py [opties]

Opties:
    --unit           Voer alleen unit tests uit
    --integration    Voer alleen integratie tests uit
    --webtest        Voer alleen web tests uit
    --all            Voer alle tests uit (standaard)
    --verbose        Toon gedetailleerde uitvoer
    --html           Genereer HTML coverage rapport
"""
import sys
import os
import subprocess
import argparse


def run_pytest(args):
    """
    Voer pytest uit met de gegeven argumenten.
    
    Args:
        args (list): Lijst met argumenten voor pytest
        
    Returns:
        int: Exit code van pytest
    """
    cmd = [sys.executable, "-m", "pytest"] + args
    print(f"Uitvoeren commando: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    """
    Hoofdfunctie voor het uitvoeren van tests.
    """
    parser = argparse.ArgumentParser(description="Voer tests uit en genereer een coverage rapport.")
    parser.add_argument("--unit", action="store_true", help="Voer alleen unit tests uit")
    parser.add_argument("--integration", action="store_true", help="Voer alleen integratie tests uit")
    parser.add_argument("--webtest", action="store_true", help="Voer alleen web tests uit")
    parser.add_argument("--all", action="store_true", help="Voer alle tests uit (standaard)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Toon gedetailleerde uitvoer")
    parser.add_argument("--html", action="store_true", help="Genereer HTML coverage rapport")
    
    args = parser.parse_args()
    
    # Als geen testtype is opgegeven, voer dan alle tests uit
    if not (args.unit or args.integration or args.webtest or args.all):
        args.all = True
    
    # Bouw de argumenten voor pytest
    pytest_args = []
    
    # Voeg markers toe op basis van de geselecteerde tests
    if args.unit:
        pytest_args.append("-m unit")
    elif args.integration:
        pytest_args.append("-m integration")
    elif args.webtest:
        pytest_args.append("-m webtest")
    
    # Voeg verbose mode toe indien gewenst
    if args.verbose:
        pytest_args.append("-v")
    
    # Voeg coverage toe
    pytest_args.append("--cov=src")
    pytest_args.append("--cov-report=term")
    
    if args.html:
        pytest_args.append("--cov-report=html")
    
    # Voer de tests uit
    result = run_pytest(pytest_args)
    
    # Geef de juiste exit code
    return result


if __name__ == "__main__":
    sys.exit(main())
