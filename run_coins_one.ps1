<#
    run_coins.ps1  –  sequential Optuna runs (PowerShell edition)
    Edit the $coins array to control which symbols and date ranges you run.
#>

# ------------------------------------------------------------------
#  Python entry-point
# ------------------------------------------------------------------
$pythonExe   = "python"                       # full path if needed
$optunaEntry = "dca_bot.scripts.one_stage_opt"       # `python -m …`

# ------------------------------------------------------------------
#  Coins + date ranges  (Symbol, Start, End)
# ------------------------------------------------------------------
$coins = @(
    @{ Symbol = "EOSUSDT"; Start = "2022-06-19"; End = "2025-05-09" },
    @{ Symbol = "SOLUSDT"; Start = "2020-09-03"; End = "2025-05-09" },
    @{ Symbol = "ETHUSDT"; Start = "2017-01-22"; End = "2025-05-09" },
    @{ Symbol = "SUIUSDT"; Start = "2023-07-01"; End = "2025-05-09" },
    @{ Symbol = "WIFUSDT"; Start = "2023-11-23"; End = "2025-05-09" },
    @{ Symbol = "PEPEUSDT"; Start = "2023-04-25"; End = "2025-05-09" },
    @{ Symbol = "1INCHUSDT"; Start = "2022-05-30"; End = "2025-05-09" },
    @{ Symbol = "BTCUSDT"; Start = "2018-01-01"; End = "2025-05-09" }
)

# ------------------------------------------------------------------
#  Flags common to every run
# ------------------------------------------------------------------
$commonFlags = @(
    "--trials",  "2000",
    "--jobs",    "1",                     # 0 = run jobs sequentially
    "--storage", "sqlite:///dca.sqlite",
    "--startup", "400",
    "--min-orders", "10",
    "-v"                                   # verbose logs
)

# ------------------------------------------------------------------
#  Loop and launch
# ------------------------------------------------------------------
foreach ($c in $coins) {
    Write-Host "========================================================="
    Write-Host "Optimising $($c.Symbol)   ($($c.Start) → $($c.End))"
    Write-Host "========================================================="

    # Build the full argument list:  symbol, start, end, then common flags
    $args = @(
        "-m", $optunaEntry,
        $c.Symbol,
        $c.Start,
        $c.End,
        "--use-sig",  "1",          # ignore indicator
        "--reopen-sec", "60"        # reopen 60 s after previous exit
    ) + $commonFlags

    & $pythonExe @args
}
