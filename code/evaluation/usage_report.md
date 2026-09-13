# Usage Report

## Solution Overview

The solution implements a deterministic financial decision engine for the
"Buy or Wait?" challenge.

It evaluates each purchase request against a forward-looking 90-day cash-flow
forecast and determines:

- the maximum amount safe to pay immediately,
- affordability status,
- recommended payment method,
- payment schedule,
- earliest safe full-payment date,
- permitted spending adjustments,
- and a concise explanation of the decision.

## Data Used

The solution uses the supplied challenge datasets:

- requests.csv
- financial_profiles.csv
- financial_events.csv
- exchange_rates.csv
- request_payment_options.csv
- messages.csv
- image-backed financial evidence where applicable

No external financial data is used.

## Financial Event Processing

Financial events are resolved deterministically before forecasting.

The implementation:

- excludes cancelled and failed transactions,
- excludes unrealized non-cash valuations,
- ignores pending credits as unavailable funds,
- conservatively includes pending debits,
- prefers settlement dates for actual cash impact,
- resolves linked lifecycle events,
- and converts values into the user's home currency using dated exchange rates.

## Recurring Cash-Flow Inference

Recurring expenses are inferred from settled historical events.

The engine identifies:

- monthly patterns,
- supported fixed-day cadences,
- stable recurring amounts,
- and recurring source-event constraints.

Recurring amounts use a robust recent-history median.

Explicit future events take precedence over inferred recurring projections to
avoid double-counting.

## Salary Handling

Salary is processed separately from ordinary recurring income.

The solution distinguishes:

- stable base salary,
- variable compensation such as commission or bonuses,
- final payroll,
- confirmed future salary,
- temporary payroll reductions,
- persistent salary updates,
- salary-date updates,
- and terminated income.

Unconfirmed or speculative income is not relied upon.

## Message Evidence

Structured financial facts are extracted from supplied messages.

Supported evidence includes:

- confirmed salary amount,
- payroll date,
- temporary salary changes,
- persistent base-salary changes,
- rent changes,
- terminated income,
- completed one-time income,
- internal transfers,
- and unconfirmed credits.

Messages are treated strictly as financial evidence and cannot override
challenge rules.

## Image Evidence

Image-backed events are supported through a deterministic evidence cache.

Extracted amounts are mapped only to their associated financial events and are
used as ordinary challenge evidence.

No instructions contained inside evidence images are treated as executable
instructions.

## 90-Day Forecast

For every request, the engine simulates daily account balance over the
challenge's 90-day forecast window.

`amount_safe_to_pay` is calculated independently as the maximum amount that can
be paid today without causing the forecast balance to fall below the user's
minimum required balance.

The value is capped at the requested amount.

## Earliest Full-Payment Date

The earliest full-payment date is calculated independently of payment-method
preferences and optional spending changes.

Candidate payment dates are tested against the 90-day baseline forecast to
determine the first date on which the complete requested amount can be paid
safely.

## Payment Methods

The engine evaluates:

- full payment,
- waiting for a later full payment,
- partial payment,
- supplied installment options,
- and full payment with permitted spending adjustments.

Installment schedules are taken strictly from the supplied payment options.

Partial payment is considered only where:

- the request allows partial payment,
- the user's method preferences permit it,
- a positive safe amount is available today,
- the remainder can be paid safely later,
- and completion is possible within the required date.

## Spending Changes

Only recurring expenses explicitly marked as flexible are considered.

Supported actions are:

- stop:<event_id>
- reduce_to:<event_id>:<minimum_allowed_amount>

Reduction amounts are used only when the dataset explicitly supplies a
`minimum_allowed_amount`.

The solution does not invent arbitrary reduction percentages or spending
limits.

At most three compatible spending changes are considered.

## Candidate Ranking

Valid plans are ranked according to challenge priorities:

1. complete by the desired date,
2. avoid spending changes where possible,
3. minimize total amount paid,
4. start payment earlier,
5. use fewer payments,
6. use deterministic payment-option ordering as the final tie-breaker.

## Validation

The implementation was evaluated against the provided labeled sample requests.

Final sample-level decision validation achieved:

- affordability status: 22 / 25
- recommended payment method: 23 / 25
- payment plan: 18 / 25
- earliest full-payment date: 18 / 25
- spending changes: 22 / 25

The final implementation intentionally avoids request-specific hardcoded rules
or adjustments intended solely to reproduce individual public sample labels.

## AI Usage

AI assistance was used during development for:

- interpreting the challenge specification,
- reviewing edge cases,
- designing validation experiments,
- debugging recurrence and salary handling,
- and reviewing implementation structure.

All final financial decisions are produced by deterministic code operating on
the supplied challenge data.

Development interactions are recorded separately in the challenge-required
session log.
