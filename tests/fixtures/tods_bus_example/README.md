# TODS bus example (reconstructed)

A one-block GTFS + TODS 2.1 feed shaped like the bus example in the TODS
specification (https://tods-transit.org/spec/examples/): block `BLOCK-A` on
route `12` makes four 50-minute revenue trips (`101`-`104`) between a garage
pull-out and pull-in, with the operator's 70-minute lunch inside the layover
between trips `102` and `103`.

The rows were rebuilt from the example's published totals, not copied from the
specification page, and made runnable with a daily calendar and the required
GTFS fields. Check them against the published example before relying on stop
IDs or names.

| Movement | From | To | Minutes |
|---|---|---|---:|
| Pull-out `PO1` | Garage `G` 09:45 | Downtown `A` 09:55 | 10 |
| Wait at Downtown | 09:55 | 10:00 | 5 |
| Trip `101` | Downtown 10:00 | Uptown `B` 10:50 | 50 |
| Layover at Uptown | 10:50 | 11:00 | 10 |
| Trip `102` | Uptown 11:00 | Downtown 11:50 | 50 |
| Layover at Downtown (lunch) | 11:50 | 13:00 | 70 |
| Trip `103` | Downtown 13:00 | Uptown 13:50 | 50 |
| Layover at Uptown | 13:50 | 14:00 | 10 |
| Trip `104` | Uptown 14:00 | Downtown 14:50 | 50 |
| Pull-in `PI1` | Downtown 14:50 | Garage 15:00 | 10 |

Totals: 200 revenue-trip minutes, 95 stationary minutes, 20 pull-out/pull-in
minutes, 315 vehicle minutes (09:45-15:00).
