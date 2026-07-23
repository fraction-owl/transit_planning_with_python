/**
 * Weather Calendar Blocker v10 — All-Tier Daylight Clamp + Twilight Outline Boxes
 *
 * UV tiers (traffic-light + very low):
 *   🌿 Very Low (daylight, UV < 1.0)  → Green  — shoulder/safe hours (sunrise–sunset)
 *   🌤️ Low      (UV 1.0 – 2.9)        → Yellow — minimal risk
 *   ⛅ Medium   (UV 3.0 – 5.9)        → Orange — apply sunscreen
 *   ☀️ High     (UV 6+)               → Red    — minimize exposure
 *   ⚠️ any tier → Gray when NOAA and Open-Meteo disagree by ≥ UV_DISAGREE_THRESHOLD
 *      (Gray replaces the old yellow confidence override, freeing yellow for the Low tier.)
 *   🌅/🌇 Twilight (TWILIGHT_MINUTES before sunrise / after sunset) → outline-only
 *      box (no fill) — small amounts of UV still present at ground level.
 *
 * UV data: CurrentUVIndex.com (NOAA) + Open-Meteo (GFS-derived)
 *          When both sources agree    → confident block
 *          When they disagree         → block with ⚠️ flag (gray override)
 * Rain data: Open-Meteo (GFS model, no API key)
 * Sun times: Open-Meteo daily sunrise/sunset — they move through the year, so
 *          they are re-fetched every run. They bound ALL UV tiers (v10) and
 *          position the twilight boxes. If the sun-times fetch fails, Very Low
 *          falls back to the old UV > 0 heuristic, Low/Med/High stay
 *          slot-aligned, and twilight boxes are skipped for that run.
 *
 * ROLLING 5-DAY WINDOW with TWICE-DAILY updates:
 * Runs at ~6:15 AM and ~12:15 PM. Each run deletes all weather blocks
 * from TODAY forward and recreates with the freshest forecast data.
 * Past blocks are preserved as historical record.
 *
 * Very Low and Low UV blocks are NOT gap-bridged (bridging across tier
 * boundaries would mislabel elevated UV as safer than it is).
 * UV_GAP_BRIDGE_MS is retained for Medium/High but defaults to 0 (disabled).
 *
 * UV uses linear interpolation for 30-min slots.
 * Rain uses flat-hold for 30-min slots.
 *
 * NOTE: Very Low slot selection is overlap-based — a slot whose 30-min
 * interval straddles sunrise or sunset is included, then clamped to the
 * exact sun time. Interior boundaries against other tiers remain
 * slot-aligned (30 min).
 *
 * v9 — Very Low edge snap: the day's first Very Low block starts exactly
 * at sunrise and the day's last ends exactly at sunset, even when the
 * data starts late/ends early. (v8 only TRIMMED overshooting edges, so
 * the ~6:15 AM run — whose NOAA "now" entry lands after sunrise — left
 * the AM block starting mid-morning.) The snap is skipped when the gap
 * exceeds UV_EDGE_SNAP_MAX_MS or contains elevated-UV data; midday dip
 * blocks are never stretched to the sun times.
 *
 * v9.1 — Solar noon split: any UV block — Very Low through High — that
 * spans solar noon (midpoint of sunrise/sunset; not on the hour) is
 * split into an AM and a PM block at solar noon. Blocks already bounded
 * at solar noon, or entirely on one side of it, are left alone, as are
 * blocks where a split would create a half shorter than one slot. Rain
 * blocks are not split. (v9 split only the Very Low tier, so on sunny
 * days — when the only noon-spanning block is Medium/High — no split
 * was visible anywhere.)
 *
 * v10 — All-tier daylight clamp: EVERY UV block — Very Low through High —
 * is now clamped to the exact sunrise/sunset of its day. (Through v9.1
 * only Very Low was clamped; Low/Med/High blocks were slot-aligned, so an
 * interpolated UV ≥ 1 slot straddling sunrise or sunset let yellow/orange
 * blocks start before sunrise or run past sunset.) The clamp is trim-only
 * for Low/Med/High — it never stretches a block, so it can't mislabel
 * where elevated UV starts and stops; the Very Low snap logic is
 * unchanged. Blocks with no daylight overlap at all (bogus dark-hours
 * "UV" from source disagreement) are dropped. Rain blocks are not
 * clamped — rain doesn't care about daylight.
 *
 * v10 — Twilight outline boxes: each day gets two extra boxes, one ending
 * exactly at sunrise (🌅 Dawn) and one starting exactly at sunset
 * (🌇 Dusk), TWILIGHT_MINUTES (default 30) long. They mark the twilight
 * shoulder where the sun is below the horizon but small amounts of
 * scattered UV still reach ground level. They render with NO FILL and
 * only an outline: Google Calendar has no native outline event style, so
 * the script marks the owner's own attendance as "needsAction" via the
 * Calendar advanced service — the calendar UI draws un-responded events
 * as outlined, unfilled boxes in the event's color. This requires the
 * Calendar advanced service (SETUP step 4); without it the boxes fall
 * back to solid green and a one-time warning is logged. Twilight boxes
 * are pure astronomy — no UV-data dependency — so they appear even when
 * both UV sources fail, and are skipped only when the sun-times fetch
 * itself fails.
 *
 * The UV > 0 fallback path (sun-times fetch failed) is slot-aligned only
 * — no trim, snap, split, or twilight boxes.
 *
 * REQUIRES: V8 runtime (set in appsscript.json or Project Settings → Runtime)
 *
 * SETUP:
 *   1. Go to https://script.google.com → New Project
 *   2. Delete the placeholder code and paste this entire file
 *   3. Also create/edit appsscript.json (View → Show manifest file)
 *   4. Enable the Calendar advanced service (needed only for the twilight
 *      outline styling): Editor → Services → + → "Google Calendar API" →
 *      Add, keeping the default identifier "Calendar". Skipping this step
 *      only costs the outline look — twilight boxes then render solid.
 *   5. Adjust LAT, LNG, and TIMEZONE below if needed (defaults to DC area)
 *   6. Save (Ctrl+S)
 *   7. Select "testSetup" → Run (▶️) — first run asks for permissions
 *   8. Select "installTriggers" → Run once to set up twice-daily auto-runs
 *   9. Optionally run "createWeatherCalendarBlocks" to test a real calendar write
 *
 * RUN DROPDOWN: Only the four entry points appear in the editor's Run menu —
 * createWeatherCalendarBlocks, installTriggers, removeAllTriggers, testSetup.
 * Every other function name ends in "_", which Apps Script treats as private
 * (hidden from the dropdown and not directly runnable). Once installTriggers
 * has been run, no manual runs are needed — the script maintains itself.
 *
 * UPGRADING from v9.x or earlier: paste this file over the old code, save,
 * enable the Calendar advanced service (SETUP step 4), and run
 * "installTriggers" once. Triggers pointing at removed legacy names
 * (e.g. createUvCalendarBlocks) are cleaned up automatically; leaving them
 * in place would cause failed-trigger emails since those functions no
 * longer exist.
 *
 * UV data credit: https://currentuvindex.com (CC BY 4.0)
 */

// ============ CONFIGURATION ============
const LAT = 38.9072;              // Your latitude (default: Washington DC area)
const LNG = -77.0369;             // Your longitude
const TIMEZONE = 'America/New_York'; // Your timezone
const FORECAST_DAYS = 5;          // Rolling forecast window
const UV_VERY_LOW_MAX = 1.0;      // Daylight, UV < 1.0 → Very Low (green)
const UV_LOW_MAX = 3.0;           // 1.0 ≤ UV < 3.0 → Low (yellow)
const UV_HIGH_THRESHOLD = 6.0;    // UV ≥ 6.0 → High (red); 3.0–5.9 → Medium (orange)
const UV_GAP_BRIDGE_MS = 0;       // Bridge Medium/High gaps ≤ this ms; 0 = disabled.
                                  // Re-enable (e.g. 60 * 60 * 1000) with caution:
                                  // bridging across tier boundaries causes overlap.
const UV_DISAGREE_THRESHOLD = 1.5; // Sources differ by 1.5+ = flag ⚠️ (gray)
const UV_EDGE_SNAP_MAX_MS = 60 * 60 * 1000; // Snap the day's first/last Very Low block
                                  // edge to exact sunrise/sunset if the gap is ≤ this
                                  // AND the gap has no elevated-UV data (i.e. the gap
                                  // is missing data, e.g. the 6:15 AM run whose NOAA
                                  // "now" entry starts after sunrise). 0 = trim-only.
const TWILIGHT_MINUTES = 30;      // Outline-only twilight box: this many minutes
                                  // ending at sunrise (🌅 Dawn) and starting at
                                  // sunset (🌇 Dusk) — small amounts of UV are
                                  // still present at ground level. 0 = disabled.
const RAIN_MODERATE_MM = 2.5;     // 2.5+ mm/hr = moderate rain
const RAIN_HEAVY_MM = 7.6;        // 7.6+ mm/hr = heavy rain
const CALENDAR_ID = 'primary';    // Use your primary Google Calendar
const TRIGGER_FUNCTION = 'createWeatherCalendarBlocks';
const SLOT_MINUTES = 30;          // Calendar block granularity

// ============ MAIN FUNCTION ============
function createWeatherCalendarBlocks() {
  try {
    // Fetch UV from both sources
    const uvNoaa = fetchUvForecastNoaa_();
    const uvOpenMeteo = fetchUvForecastOpenMeteo_();

    // Fetch rain from Open-Meteo
    const rainHourly = fetchRainForecast_();

    // Fetch sunrise/sunset — bounds every UV tier and places twilight boxes
    const sunTimes = fetchSunTimes_();

    if (!uvNoaa && !uvOpenMeteo && !rainHourly) {
      Logger.log('All data sources failed — skipping this run.');
      return;
    }

    // Merge UV sources: use NOAA as primary, compare with Open-Meteo for confidence
    const uvMerged = mergeUvSources_(uvNoaa, uvOpenMeteo);

    // Remove weather blocks from TODAY forward
    removeWeatherBlocksFromTodayForward_();

    // Build and create UV blocks (all four tiers)
    let uvCount = 0;
    if (uvMerged?.length) {
      const uvSlots = interpolateUvToSlots_(uvMerged);

      // Medium/High blocks (UV ≥ UV_LOW_MAX) — bridging controlled by UV_GAP_BRIDGE_MS
      const medHighRaw = buildMedHighUvBlocks_(uvSlots);
      const medHighBridged = bridgeGaps_(medHighRaw, UV_GAP_BRIDGE_MS);
      const bridged = medHighRaw.length - medHighBridged.length;
      const medHighClamped = trimToDaylight_(medHighBridged, sunTimes, uvSlots, 'Med/High UV');
      const medHighBlocks = splitAtSolarNoon_(medHighClamped, sunTimes, uvSlots);

      // Low blocks (UV_VERY_LOW_MAX ≤ UV < UV_LOW_MAX) — not bridged
      const lowClamped = trimToDaylight_(buildLowUvBlocks_(uvSlots), sunTimes, uvSlots, 'Low UV');
      const lowBlocks = splitAtSolarNoon_(lowClamped, sunTimes, uvSlots);

      // Very Low blocks (daylight, UV < UV_VERY_LOW_MAX) — not bridged;
      // clamped (trim + snap) and split internally
      const veryLowBlocks = buildVeryLowUvBlocks_(uvSlots, sunTimes);

      for (const block of medHighBlocks) createUvCalendarBlock_(block);
      for (const block of lowBlocks) createUvCalendarBlock_(block);
      for (const block of veryLowBlocks) createUvCalendarBlock_(block);

      uvCount = medHighBlocks.length + lowBlocks.length + veryLowBlocks.length;
      const allUvBlocks = [...medHighBlocks, ...lowBlocks, ...veryLowBlocks];
      const flagged = allUvBlocks.filter(b => b.lowConfidence).length;

      Logger.log(`Created ${uvCount} UV block(s)`
        + ` (${medHighBlocks.length} med/high, ${lowBlocks.length} low, ${veryLowBlocks.length} very low)`
        + (bridged > 0 ? `, ${bridged} gap(s) bridged` : '')
        + (flagged > 0 ? `, ${flagged} flagged ⚠️` : '')
        + '.');
    } else {
      Logger.log('No UV data available — skipping UV blocks.');
    }

    // Build and create twilight boxes (outline-only, pure astronomy)
    let twilightCount = 0;
    if (TWILIGHT_MINUTES > 0) {
      if (sunTimes?.length) {
        const twilightBoxes = buildTwilightBoxes_(sunTimes);
        for (const box of twilightBoxes) createTwilightCalendarBlock_(box);
        twilightCount = twilightBoxes.length;
        Logger.log(`Created ${twilightCount} twilight box(es).`);
      } else {
        Logger.log('Sun times unavailable — skipping twilight boxes.');
      }
    }

    // Build and create rain blocks
    let rainCount = 0;
    if (rainHourly?.length) {
      const rainSlots = flatExpandToSlots_(rainHourly);
      const rainBlocks = buildRainBlocks_(rainSlots);
      for (const block of rainBlocks) {
        createRainCalendarBlock_(block);
      }
      rainCount = rainBlocks.length;
      Logger.log(`Created ${rainCount} rain block(s).`);
    } else {
      Logger.log('No rain data available — skipping rain blocks.');
    }

    Logger.log(`Total: ${uvCount + twilightCount + rainCount} weather block(s) across ${FORECAST_DAYS} days.`);

  } catch (e) {
    Logger.log(`❌ Top-level error: ${e.message}`);
    Logger.log(e.stack);
  }
}

// ============ UV SOURCE MERGING ============
/**
 * Merges NOAA and Open-Meteo UV forecasts.
 *
 * Strategy:
 * - NOAA is primary (more direct UV modeling from NOAA data)
 * - Open-Meteo is comparison source (GFS-derived approximation)
 * - For each hour, we use NOAA's value but flag low confidence
 *   when the two sources disagree by ≥ UV_DISAGREE_THRESHOLD
 * - If NOAA fails, fall back to Open-Meteo entirely (all flagged)
 * - If Open-Meteo fails, use NOAA without confidence info
 *
 * Returns array of { time, uv, lowConfidence, noaaUv, openMeteoUv }
 */
function mergeUvSources_(noaaData, openMeteoData) {
  // Both failed
  if (!noaaData?.length && !openMeteoData?.length) return null;

  // Only Open-Meteo available — use it but flag everything
  if (!noaaData?.length && openMeteoData?.length) {
    Logger.log('⚠️ NOAA UV unavailable — using Open-Meteo only (all blocks flagged).');
    return openMeteoData.map(e => ({
      time: e.time,
      uv: e.uv,
      lowConfidence: true,
      noaaUv: null,
      openMeteoUv: e.uv
    }));
  }

  // Only NOAA available — use it without confidence info
  if (noaaData?.length && !openMeteoData?.length) {
    Logger.log('Open-Meteo UV unavailable — using NOAA only (no confidence comparison).');
    return noaaData.map(e => ({
      time: e.time,
      uv: e.uv,
      lowConfidence: false,
      noaaUv: e.uv,
      openMeteoUv: null
    }));
  }

  // Both available — merge by matching timestamps
  const openMeteoMap = new Map();
  for (const e of openMeteoData) {
    const hourKey = new Date(e.time);
    hourKey.setMinutes(0, 0, 0);
    openMeteoMap.set(hourKey.getTime(), e.uv);
  }

  let agreeCount = 0;
  let disagreeCount = 0;

  const merged = noaaData.map(e => {
    const hourKey = new Date(e.time);
    hourKey.setMinutes(0, 0, 0);
    const omUv = openMeteoMap.get(hourKey.getTime()) ?? null;

    let lowConfidence = false;
    if (omUv !== null) {
      const diff = Math.abs(e.uv - omUv);
      lowConfidence = diff >= UV_DISAGREE_THRESHOLD;
      if (lowConfidence) {
        disagreeCount++;
      } else {
        agreeCount++;
      }
    }

    return {
      time: e.time,
      uv: e.uv,
      lowConfidence,
      noaaUv: e.uv,
      openMeteoUv: omUv
    };
  });

  const total = agreeCount + disagreeCount;
  if (total > 0) {
    const pct = Math.round((agreeCount / total) * 100);
    Logger.log(`UV source agreement: ${agreeCount}/${total} hours agree (${pct}%), ${disagreeCount} flagged.`);
  }

  return merged;
}

// ============ CURRENTUVINDEX.COM API (NOAA UV) ============
function fetchUvForecastNoaa_() {
  const url = `https://currentuvindex.com/api/v1/uvi?latitude=${LAT}&longitude=${LNG}`;
  const MAX_RETRIES = 3;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const response = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
      const code = response.getResponseCode();

      if (code === 200) {
        const data = JSON.parse(response.getContentText());

        if (!data.ok) {
          Logger.log(`NOAA UV API error: ${data.message}`);
          return null;
        }

        const entries = [];

        if (data.now) {
          entries.push({ time: new Date(data.now.time), uv: data.now.uvi ?? 0 });
        }

        if (data.forecast?.length) {
          for (const point of data.forecast) {
            entries.push({ time: new Date(point.time), uv: point.uvi ?? 0 });
          }
        }

        entries.sort((a, b) => a.time - b.time);
        const deduped = [];
        for (const e of entries) {
          if (deduped.length === 0 || e.time.getTime() !== deduped[deduped.length - 1].time.getTime()) {
            deduped.push(e);
          }
        }

        const { windowStart, windowEnd } = getForecastWindow_();
        const filtered = deduped.filter(e => e.time >= windowStart && e.time < windowEnd);

        Logger.log(`NOAA UV: ${filtered.length} hourly entries.`);
        return filtered;
      }

      Logger.log(`NOAA UV HTTP ${code} (attempt ${attempt}/${MAX_RETRIES})`);
      if (code >= 400 && code < 500) return null;

    } catch (e) {
      Logger.log(`NOAA UV fetch error (attempt ${attempt}/${MAX_RETRIES}): ${e.message}`);
    }

    if (attempt < MAX_RETRIES) Utilities.sleep(Math.pow(2, attempt) * 1000);
  }

  Logger.log('All NOAA UV fetch attempts failed.');
  return null;
}

// ============ OPEN-METEO API (COMPARISON UV + RAIN + SUN TIMES) ============
/**
 * Fetches UV from Open-Meteo (GFS-derived) for comparison purposes.
 */
function fetchUvForecastOpenMeteo_() {
  const params = [
    `latitude=${LAT}`,
    `longitude=${LNG}`,
    'hourly=uv_index',
    `timezone=${encodeURIComponent(TIMEZONE)}`,
    `forecast_days=${FORECAST_DAYS}`
  ].join('&');

  const url = `https://api.open-meteo.com/v1/forecast?${params}`;
  const MAX_RETRIES = 3;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const response = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
      const code = response.getResponseCode();

      if (code === 200) {
        const { hourly } = JSON.parse(response.getContentText());

        if (!hourly?.time) {
          Logger.log('No UV data in Open-Meteo response.');
          return null;
        }

        const entries = hourly.time.map((timeStr, i) => ({
          time: new Date(timeStr),
          uv: hourly.uv_index?.[i] ?? 0
        }));

        Logger.log(`Open-Meteo UV: ${entries.length} hourly entries.`);
        return entries;
      }

      Logger.log(`Open-Meteo UV HTTP ${code} (attempt ${attempt}/${MAX_RETRIES})`);
      if (code >= 400 && code < 500) return null;

    } catch (e) {
      Logger.log(`Open-Meteo UV fetch error (attempt ${attempt}/${MAX_RETRIES}): ${e.message}`);
    }

    if (attempt < MAX_RETRIES) Utilities.sleep(Math.pow(2, attempt) * 1000);
  }

  Logger.log('All Open-Meteo UV fetch attempts failed.');
  return null;
}

/**
 * Fetches hourly rain forecast from Open-Meteo.
 */
function fetchRainForecast_() {
  const params = [
    `latitude=${LAT}`,
    `longitude=${LNG}`,
    'hourly=precipitation_probability,precipitation,weather_code',
    'temperature_unit=fahrenheit',
    `timezone=${encodeURIComponent(TIMEZONE)}`,
    `forecast_days=${FORECAST_DAYS}`
  ].join('&');

  const url = `https://api.open-meteo.com/v1/forecast?${params}`;
  const MAX_RETRIES = 3;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const response = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
      const code = response.getResponseCode();

      if (code === 200) {
        const { hourly } = JSON.parse(response.getContentText());

        if (!hourly?.time) {
          Logger.log('No rain data in Open-Meteo response.');
          return null;
        }

        const entries = hourly.time.map((timeStr, i) => ({
          time: new Date(timeStr),
          precipitation: hourly.precipitation?.[i] ?? 0,
          rainChance: hourly.precipitation_probability?.[i] ?? null,
          weatherCode: hourly.weather_code?.[i] ?? null
        }));

        Logger.log(`Rain forecast: ${entries.length} hourly entries.`);
        return entries;
      }

      Logger.log(`Rain HTTP ${code} (attempt ${attempt}/${MAX_RETRIES})`);
      if (code >= 400 && code < 500) return null;

    } catch (e) {
      Logger.log(`Rain fetch error (attempt ${attempt}/${MAX_RETRIES}): ${e.message}`);
    }

    if (attempt < MAX_RETRIES) Utilities.sleep(Math.pow(2, attempt) * 1000);
  }

  Logger.log('All rain fetch attempts failed.');
  return null;
}

/**
 * Fetches daily sunrise/sunset times from Open-Meteo.
 * Bounds every UV tier and positions the twilight boxes. Sunrise/sunset
 * move through the year, so this is re-fetched on every run — the blocks
 * always follow the current sun times, never a hardcoded clock time.
 *
 * NOTE: With the timezone param set, Open-Meteo returns local ISO strings
 * without an offset; new Date() parses them in the Apps Script project's
 * timezone. This matches the existing Open-Meteo UV/rain parsing, so the
 * project-timezone check in testSetup applies here too.
 *
 * solarNoon is computed as the midpoint of sunrise and sunset (Open-Meteo
 * has no solar-transit daily variable). The midpoint tracks true solar
 * transit to within about a minute, which is well inside slot granularity.
 *
 * Returns array of { sunrise, sunset, solarNoon } Dates, or null on failure.
 */
function fetchSunTimes_() {
  const params = [
    `latitude=${LAT}`,
    `longitude=${LNG}`,
    'daily=sunrise,sunset',
    `timezone=${encodeURIComponent(TIMEZONE)}`,
    `forecast_days=${FORECAST_DAYS}`
  ].join('&');

  const url = `https://api.open-meteo.com/v1/forecast?${params}`;
  const MAX_RETRIES = 3;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const response = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
      const code = response.getResponseCode();

      if (code === 200) {
        const { daily } = JSON.parse(response.getContentText());

        if (!daily?.sunrise?.length) {
          Logger.log('No sunrise/sunset data in Open-Meteo response.');
          return null;
        }

        const entries = daily.sunrise.map((s, i) => {
          const sunrise = new Date(s);
          const sunset = new Date(daily.sunset[i]);
          return {
            sunrise,
            sunset,
            solarNoon: new Date((sunrise.getTime() + sunset.getTime()) / 2)
          };
        });

        Logger.log(`Sun times: ${entries.length} day(s).`);
        return entries;
      }

      Logger.log(`Sun times HTTP ${code} (attempt ${attempt}/${MAX_RETRIES})`);
      if (code >= 400 && code < 500) return null;

    } catch (e) {
      Logger.log(`Sun times fetch error (attempt ${attempt}/${MAX_RETRIES}): ${e.message}`);
    }

    if (attempt < MAX_RETRIES) Utilities.sleep(Math.pow(2, attempt) * 1000);
  }

  Logger.log('All sun times fetch attempts failed.');
  return null;
}

/**
 * Returns true if the 30-min slot starting at `slotStart` overlaps any
 * daylight period. Testing the slot interval (rather than its start
 * instant) means a slot straddling sunrise or sunset is included, so
 * clampToDaylight_ can trim it to the exact sun time instead of the
 * whole slot being excluded. Called only at runtime, after top-level
 * consts (SLOT_MS) are initialized.
 */
function overlapsDaylight_(slotStart, sunTimes) {
  const slotEnd = new Date(slotStart.getTime() + SLOT_MS);
  return sunTimes.some(d => slotStart < d.sunset && slotEnd > d.sunrise);
}

// ============ SLOT EXPANSION ============
/**
 * Interpolates hourly UV readings into 30-minute slots.
 * Carries forward the lowConfidence flag from the source hours.
 */
function interpolateUvToSlots_(hourlyData) {
  const slots = [];
  const slotMs = SLOT_MINUTES * 60 * 1000;

  for (let i = 0; i < hourlyData.length; i++) {
    const current = hourlyData[i];
    const next = hourlyData[i + 1] ?? null;

    // :00 slot — exact reading
    slots.push({
      time: current.time,
      uv: current.uv,
      lowConfidence: current.lowConfidence ?? false
    });

    // :30 slot — interpolated
    const midpointUv = next ? (current.uv + next.uv) / 2 : current.uv;
    const midpointConfidence = (current.lowConfidence ?? false) || (next?.lowConfidence ?? false);

    slots.push({
      time: new Date(current.time.getTime() + slotMs),
      uv: Math.round(midpointUv * 10) / 10,
      lowConfidence: midpointConfidence
    });
  }

  return slots;
}

/**
 * Flat-hold expansion for rain.
 */
function flatExpandToSlots_(hourlyData) {
  const slots = [];
  const slotMs = SLOT_MINUTES * 60 * 1000;
  const slotsPerHour = 60 / SLOT_MINUTES;

  for (const entry of hourlyData) {
    for (let i = 0; i < slotsPerHour; i++) {
      slots.push({
        ...entry,
        time: new Date(entry.time.getTime() + i * slotMs)
      });
    }
  }

  return slots;
}

// ============ TRIGGER MANAGEMENT ============
/**
 * Installs twice-daily triggers: ~6:15 AM and ~12:15 PM.
 * Safe to re-run — removes old triggers first.
 */
function installTriggers() {
  const targetFunctions = [TRIGGER_FUNCTION, 'createUvCalendarBlocks']; // legacy handler name (≤ v7) kept so old triggers get cleaned up
  const existing = ScriptApp.getProjectTriggers()
    .filter(t => targetFunctions.includes(t.getHandlerFunction()));

  for (const trigger of existing) {
    ScriptApp.deleteTrigger(trigger);
    Logger.log(`Removed old trigger: ${trigger.getHandlerFunction()} (${trigger.getUniqueId()})`);
  }

  // Morning run — catches overnight forecast updates
  ScriptApp.newTrigger(TRIGGER_FUNCTION)
    .timeBased()
    .everyDays(1)
    .atHour(6)
    .nearMinute(15)
    .inTimezone(TIMEZONE)
    .create();

  // Midday run — catches morning forecast revisions for afternoon/evening
  ScriptApp.newTrigger(TRIGGER_FUNCTION)
    .timeBased()
    .everyDays(1)
    .atHour(12)
    .nearMinute(15)
    .inTimezone(TIMEZONE)
    .create();

  Logger.log(`✅ Twice-daily triggers installed:`);
  Logger.log(`   ~6:15 AM ${TIMEZONE} (morning forecast)`);
  Logger.log(`   ~12:15 PM ${TIMEZONE} (midday revision)`);
}

function removeAllTriggers() {
  const triggers = ScriptApp.getProjectTriggers();
  for (const trigger of triggers) {
    ScriptApp.deleteTrigger(trigger);
  }
  Logger.log(`Removed ${triggers.length} trigger(s).`);
}

// ============ BLOCK BUILDING ============
const SLOT_MS = SLOT_MINUTES * 60 * 1000;

function buildConsecutiveBlocks_(slotData, filterFn, classifyFn, peakField) {
  const relevant = slotData
    .filter(filterFn)
    .sort((a, b) => a.time - b.time);

  if (relevant.length === 0) return [];

  const blocks = [];
  let current = null;

  for (const entry of relevant) {
    const severity = classifyFn(entry);

    if (current === null) {
      current = {
        start: entry.time,
        end: new Date(entry.time.getTime() + SLOT_MS),
        peakValue: entry[peakField],
        severity,
        lowConfidence: entry.lowConfidence ?? false
      };
    } else {
      const gap = entry.time.getTime() - current.end.getTime();

      if (gap <= 0 && severity === current.severity) {
        current.end = new Date(entry.time.getTime() + SLOT_MS);
        current.peakValue = Math.max(current.peakValue, entry[peakField]);
        // Block is low confidence if ANY slot in it is low confidence
        if (entry.lowConfidence) current.lowConfidence = true;
      } else {
        blocks.push(current);
        current = {
          start: entry.time,
          end: new Date(entry.time.getTime() + SLOT_MS),
          peakValue: entry[peakField],
          severity,
          lowConfidence: entry.lowConfidence ?? false
        };
      }
    }
  }

  if (current) blocks.push(current);
  return blocks;
}

/**
 * Merges consecutive same-severity blocks separated by ≤ maxGapMs.
 * UV_GAP_BRIDGE_MS = 0 makes this a no-op for UV (safe default).
 */
function bridgeGaps_(blocks, maxGapMs) {
  if (blocks.length <= 1) return blocks;

  const merged = [{ ...blocks[0] }];

  for (let i = 1; i < blocks.length; i++) {
    const prev = merged[merged.length - 1];
    const curr = blocks[i];
    const gap = curr.start.getTime() - prev.end.getTime();

    if (gap <= maxGapMs && gap >= 0 && prev.severity === curr.severity) {
      prev.end = curr.end;
      prev.peakValue = Math.max(prev.peakValue, curr.peakValue);
      if (curr.lowConfidence) prev.lowConfidence = true;
    } else {
      merged.push({ ...curr });
    }
  }

  return merged;
}

/**
 * Builds Medium and High UV blocks (UV ≥ UV_LOW_MAX).
 * These are the only blocks eligible for gap bridging.
 * The caller clamps the result to daylight via trimToDaylight_ (v10).
 */
function buildMedHighUvBlocks_(slotData) {
  return buildConsecutiveBlocks_(
    slotData,
    entry => entry.uv >= UV_LOW_MAX,
    entry => entry.uv >= UV_HIGH_THRESHOLD ? 'High' : 'Medium',
    'uv'
  );
}

/**
 * Builds Low UV blocks (UV_VERY_LOW_MAX ≤ UV < UV_LOW_MAX).
 * Not bridged — bridging would absorb Very Low territory into Low blocks.
 * The caller clamps the result to daylight via trimToDaylight_ (v10).
 */
function buildLowUvBlocks_(slotData) {
  return buildConsecutiveBlocks_(
    slotData,
    entry => entry.uv >= UV_VERY_LOW_MAX && entry.uv < UV_LOW_MAX,
    () => 'Low',
    'uv'
  );
}

/**
 * Builds Very Low UV (shoulder) blocks.
 *
 * Primary path: any slot whose 30-min interval overlaps daylight (per
 * Open-Meteo daily sun times) with UV < UV_VERY_LOW_MAX. Overlap-based
 * selection captures slots that straddle sunrise or sunset; clampToDaylight_
 * then trims them to the exact sun times. This means overcast days where
 * UV never reaches the Low threshold can produce a single Very Low block
 * spanning the whole daylight period.
 *
 * Fallback path (sun-times fetch failed): the old heuristic — UV strictly
 * between 0 and UV_VERY_LOW_MAX. The `entry.uv > 0` guard there excludes
 * overnight hours where both APIs report UV = 0, so only genuine dawn/dusk
 * shoulder ramps are captured.
 *
 * Edge handling (primary path):
 *   1. Trim — edges overshooting sunrise/sunset are clamped to the sun time.
 *   2. Snap — the day's FIRST block starts exactly at sunrise and the day's
 *      LAST block ends exactly at sunset, even when the underlying data
 *      starts late / ends early (the 6:15 AM trigger run is the usual
 *      culprit: NOAA's "now" entry lands after sunrise, so trim alone left
 *      the AM block starting at ~6:16). The snap is skipped ("almost") when
 *      the gap exceeds UV_EDGE_SNAP_MAX_MS or when any slot in the gap has
 *      UV ≥ UV_VERY_LOW_MAX — an elevated-UV gap means the block genuinely
 *      doesn't reach the sun time, and stretching it would overlap a
 *      Low/Med/High block and mislabel elevated UV as safe.
 *   3. Split — a block spanning solar noon is split into an AM and a PM
 *      block at solar noon (see splitAtSolarNoon_).
 * The fallback path (sun-times fetch failed) is slot-aligned only — no
 * trim, snap, or split, since there are no sun times to work against.
 * Not bridged.
 */
function buildVeryLowUvBlocks_(slotData, sunTimes) {
  const filterFn = sunTimes?.length
    ? entry => overlapsDaylight_(entry.time, sunTimes) && entry.uv < UV_VERY_LOW_MAX
    : entry => entry.uv > 0 && entry.uv < UV_VERY_LOW_MAX; // fallback

  if (!sunTimes?.length) {
    Logger.log('⚠️ Sun times unavailable — Very Low tier using UV > 0 fallback.');
  }

  const blocks = buildConsecutiveBlocks_(slotData, filterFn, () => 'Very Low', 'uv');
  if (!sunTimes?.length) return blocks;

  const clamped = clampToDaylight_(blocks, sunTimes, slotData);
  return splitAtSolarNoon_(clamped, sunTimes, slotData);
}

/**
 * Clamps block edges to exact sunrise/sunset, working one daylight period
 * at a time. Two operations per day:
 *
 * TRIM: edge slots straddling sunrise/sunset are cut back to the exact
 * sun time (unchanged from v8). A block with no daylight overlap is
 * dropped (shouldn't happen — every slot passed overlapsDaylight_).
 *
 * SNAP: the day's first block is extended back to start at sunrise, and
 * the day's last block extended forward to end at sunset, IF the gap is
 * ≤ UV_EDGE_SNAP_MAX_MS and contains no elevated-UV slot. Trim alone
 * could only shrink blocks, so a run whose data began after sunrise
 * (the 6:15 AM trigger) produced an AM block starting mid-morning. The
 * elevated-UV guard keeps midday dip blocks in place: a gap between
 * sunrise and a 10 AM dip block is real Low/Med UV, not missing data,
 * and must not be painted green.
 */
function clampToDaylight_(blocks, sunTimes, slotData) {
  const out = [];

  for (const day of sunTimes) {
    const dayBlocks = blocks
      .filter(b => b.start < day.sunset && b.end > day.sunrise)
      .sort((a, b) => a.start - b.start)
      .map(b => ({
        ...b,
        start: b.start < day.sunrise ? day.sunrise : b.start,
        end: b.end > day.sunset ? day.sunset : b.end
      }))
      .filter(b => b.start < b.end);

    if (dayBlocks.length === 0) continue;

    const first = dayBlocks[0];
    const sunriseGap = first.start.getTime() - day.sunrise.getTime();
    if (sunriseGap > 0 && sunriseGap <= UV_EDGE_SNAP_MAX_MS
        && !gapHasElevatedUv_(slotData, day.sunrise, first.start)) {
      first.start = day.sunrise;
    }

    const last = dayBlocks[dayBlocks.length - 1];
    const sunsetGap = day.sunset.getTime() - last.end.getTime();
    if (sunsetGap > 0 && sunsetGap <= UV_EDGE_SNAP_MAX_MS
        && !gapHasElevatedUv_(slotData, last.end, day.sunset)) {
      last.end = day.sunset;
    }

    out.push(...dayBlocks);
  }

  return out;
}

/**
 * Trims UV blocks of ANY tier to exact sunrise/sunset (v10). Each block
 * is clamped to the daylight window of the day it overlaps; a block with
 * no daylight overlap at all (dark-hours "UV" from slot alignment or
 * source disagreement) is dropped and counted in the log.
 *
 * Unlike clampToDaylight_ (Very Low only) there is no snap — Low/Med/High
 * edges must reflect where elevated UV actually starts and stops, so only
 * their overshoot past the sun times is removed, never added. Peak and
 * confidence are recomputed for trimmed blocks in case an edge slot was
 * cut away entirely. No-op when sun times are unavailable (the fallback
 * path stays slot-aligned).
 */
function trimToDaylight_(blocks, sunTimes, slotData, label) {
  if (!sunTimes?.length || !blocks.length) return blocks;

  const out = [];
  for (const block of blocks) {
    const day = sunTimes.find(d => block.start < d.sunset && block.end > d.sunrise);
    if (!day) continue;

    const trimmed = {
      ...block,
      start: block.start < day.sunrise ? day.sunrise : block.start,
      end: block.end > day.sunset ? day.sunset : block.end
    };
    const changed = trimmed.start !== block.start || trimmed.end !== block.end;
    out.push(changed ? refreshBlockStats_(trimmed, slotData) : trimmed);
  }

  const dropped = blocks.length - out.length;
  if (dropped > 0) {
    Logger.log(`${label}: dropped ${dropped} block(s) entirely outside daylight.`);
  }

  return out;
}

/**
 * True if any slot overlapping (gapStart, gapEnd) has UV at or above the
 * Very Low ceiling. Used to distinguish a missing-data gap (snap OK) from
 * an elevated-UV gap (snap forbidden).
 */
function gapHasElevatedUv_(slotData, gapStart, gapEnd) {
  return slotData.some(s => {
    const sEnd = new Date(s.time.getTime() + SLOT_MS);
    return s.time < gapEnd && sEnd > gapStart && s.uv >= UV_VERY_LOW_MAX;
  });
}

/**
 * Splits any UV block (any tier) that spans solar noon into an AM block
 * ending at solar noon and a PM block starting there. Blocks whose
 * boundary already coincides with solar noon (or that sit entirely on
 * one side) pass through untouched. To avoid sliver events, the split is
 * skipped when either half would be shorter than one slot — the block
 * then stays whole. Each half's peak UV and confidence flag are
 * recomputed from the slots it actually covers, falling back to the
 * parent block's values. Severity is inherited unchanged, which is safe:
 * buildConsecutiveBlocks_ breaks blocks on severity change, so every
 * slot in the parent shares the halves' severity. No-op when sun times
 * are unavailable.
 */
function splitAtSolarNoon_(blocks, sunTimes, slotData) {
  if (!sunTimes?.length) return blocks;

  const out = [];

  for (const block of blocks) {
    const day = sunTimes.find(d => block.start < d.sunset && block.end > d.sunrise);
    const noon = day?.solarNoon;

    const spansNoon = noon && block.start < noon && noon < block.end;
    const bothHalvesViable = spansNoon
      && (noon.getTime() - block.start.getTime() >= SLOT_MS)
      && (block.end.getTime() - noon.getTime() >= SLOT_MS);

    if (!bothHalvesViable) {
      out.push(block);
      continue;
    }

    out.push(refreshBlockStats_({ ...block, end: noon }, slotData));
    out.push(refreshBlockStats_({ ...block, start: noon }, slotData));
  }

  return out;
}

/**
 * Recomputes peakValue and lowConfidence for a (split) block from the
 * slots overlapping its interval. Keeps the existing values if no slots
 * overlap (possible on a snapped-in edge region).
 */
function refreshBlockStats_(block, slotData) {
  const covered = (slotData ?? []).filter(s => {
    const sEnd = new Date(s.time.getTime() + SLOT_MS);
    return s.time < block.end && sEnd > block.start;
  });

  if (covered.length) {
    block.peakValue = covered.reduce((max, s) => Math.max(max, s.uv), 0);
    block.lowConfidence = covered.some(s => s.lowConfidence);
  }

  return block;
}

/**
 * Builds the twilight boxes (v10): one box ending exactly at sunrise
 * (🌅 Dawn) and one starting exactly at sunset (🌇 Dusk) for each
 * forecast day, TWILIGHT_MINUTES long. Pure astronomy — built from the
 * sun times alone, no UV data involved — so the boxes track sunrise and
 * sunset as they move through the year. They abut the clamped UV blocks
 * with no gap and no overlap: UV blocks end at sunset, the Dusk box
 * starts there.
 *
 * Returns array of { kind: 'Dawn'|'Dusk', start, end }.
 */
function buildTwilightBoxes_(sunTimes) {
  if (!sunTimes?.length || TWILIGHT_MINUTES <= 0) return [];

  const twilightMs = TWILIGHT_MINUTES * 60 * 1000;
  const boxes = [];

  for (const day of sunTimes) {
    boxes.push({
      kind: 'Dawn',
      start: new Date(day.sunrise.getTime() - twilightMs),
      end: new Date(day.sunrise.getTime())
    });
    boxes.push({
      kind: 'Dusk',
      start: new Date(day.sunset.getTime()),
      end: new Date(day.sunset.getTime() + twilightMs)
    });
  }

  return boxes;
}

/**
 * Builds rain blocks.
 */
function buildRainBlocks_(slotData) {
  return buildConsecutiveBlocks_(
    slotData,
    entry => entry.precipitation >= RAIN_MODERATE_MM,
    entry => entry.precipitation >= RAIN_HEAVY_MM ? 'Heavy' : 'Moderate',
    'precipitation'
  );
}

// ============ CALENDAR OPERATIONS ============
function getForecastWindow_() {
  const windowStart = new Date();
  windowStart.setHours(0, 0, 0, 0);
  const windowEnd = new Date(windowStart);
  windowEnd.setDate(windowEnd.getDate() + FORECAST_DAYS);
  return { windowStart, windowEnd };
}

function removeWeatherBlocksFromTodayForward_() {
  const calendar = CalendarApp.getCalendarById(CALENDAR_ID);
  const { windowStart, windowEnd } = getForecastWindow_();

  const events = calendar.getEvents(windowStart, windowEnd);
  let removed = 0;

  for (const event of events) {
    const title = event.getTitle();
    let isWeatherBlock = false;

    try {
      isWeatherBlock = event.getTag('weather-blocker') === 'true' ||
                       event.getTag('uv-blocker') === 'true';
    } catch (e) {}

    if (!isWeatherBlock) {
      isWeatherBlock =
        /^(☀️ UV|⚠️☀️ UV|⛅ UV|⚠️⛅ UV|🌤️ UV|⚠️🌤️ UV|🌿 UV|⚠️🌿 UV|🌅 Dawn UV|🌇 Dusk UV|🌧️ Heavy Rain|🌦️ Rain)/.test(title) ||
        /UV (High|Medium|Low|Very Low)/.test(title) ||
        /(Heavy )?Rain \(/.test(title);
    }

    if (isWeatherBlock) {
      event.deleteEvent();
      removed++;
    }
  }

  if (removed > 0) {
    Logger.log(`Removed ${removed} existing weather block(s).`);
  }
}

function createUvCalendarBlock_(block) {
  const calendar = CalendarApp.getCalendarById(CALENDAR_ID);
  const rounded = Math.round(block.peakValue * 10) / 10;
  const flag = block.lowConfidence ? '⚠️' : '';

  let title, advice;
  if (block.severity === 'High') {
    title = `${flag}☀️ UV High (peak ${rounded})`;
    advice = 'High UV — wear sunscreen and minimize direct sun exposure.';
  } else if (block.severity === 'Medium') {
    title = `${flag}⛅ UV Medium (peak ${rounded})`;
    advice = 'Moderate UV — apply sunscreen.';
  } else if (block.severity === 'Low') {
    title = `${flag}🌤️ UV Low (peak ${rounded})`;
    advice = 'Low UV — sunscreen optional for brief exposure.';
  } else {
    // Very Low — daylight shoulder hours
    title = `${flag}🌿 UV Very Low (peak ${rounded})`;
    advice = 'Very low UV — ideal window for outdoor activity.';
  }

  const confidenceNote = block.lowConfidence
    ? 'NOAA and Open-Meteo disagree on UV for this period — actual conditions may differ.'
    : 'NOAA and Open-Meteo agree on UV for this period.';

  const description = [
    `UV severity: ${block.severity}`,
    `Peak UV index: ${rounded}`,
    '',
    confidenceNote,
    '',
    'Auto-generated by Weather Calendar Blocker.',
    'Primary UV source: currentuvindex.com (NOAA).',
    `Comparison: Open-Meteo (GFS). ${advice}`
  ].join('\n');

  const event = calendar.createEvent(title, block.start, block.end, { description });
  event.setTransparency(CalendarApp.EventTransparency.TRANSPARENT);

  if (block.lowConfidence) {
    // Gray = uncertain. All four tier colors are occupied (green/yellow/orange/red),
    // so gray is the unambiguous "we're not sure" signal. The ⚠️ in the title reinforces it.
    event.setColor(CalendarApp.EventColor.GRAY);
  } else if (block.severity === 'High') {
    event.setColor(CalendarApp.EventColor.RED);
  } else if (block.severity === 'Medium') {
    event.setColor(CalendarApp.EventColor.ORANGE);
  } else if (block.severity === 'Low') {
    event.setColor(CalendarApp.EventColor.YELLOW);
  } else {
    // Very Low
    event.setColor(CalendarApp.EventColor.GREEN);
  }

  try {
    event.setTag('weather-blocker', 'true');
  } catch (e) {}

  Logger.log(`Created: ${title} | ${fmtDate_(block.start)} ${fmtTime_(block.start)}–${fmtTime_(block.end)}`);
}

/**
 * Creates a twilight box (v10) — the TWILIGHT_MINUTES before sunrise or
 * after sunset when the sun is below the horizon but small amounts of
 * scattered UV still reach ground level. Rendered outline-only (no fill)
 * via applyOutlineStyle_; the outline inherits the event color (green —
 * even less UV than the solid-green Very Low tier).
 */
function createTwilightCalendarBlock_(box) {
  const calendar = CalendarApp.getCalendarById(CALENDAR_ID);
  const isDawn = box.kind === 'Dawn';
  const title = isDawn ? '🌅 Dawn UV (trace)' : '🌇 Dusk UV (trace)';

  const description = [
    `Twilight — ${TWILIGHT_MINUTES} min ${isDawn ? 'before sunrise' : 'after sunset'}`,
    isDawn ? `Sunrise: ${fmtTime_(box.end)}` : `Sunset: ${fmtTime_(box.start)}`,
    '',
    'The sun is below the horizon, but small amounts of scattered UV are',
    'still present at ground level.',
    '',
    'Auto-generated by Weather Calendar Blocker.',
    'Sun times: Open-Meteo.'
  ].join('\n');

  const event = calendar.createEvent(title, box.start, box.end, { description });
  event.setTransparency(CalendarApp.EventTransparency.TRANSPARENT);
  event.setColor(CalendarApp.EventColor.GREEN);

  try {
    event.setTag('weather-blocker', 'true');
  } catch (e) {}

  applyOutlineStyle_(event);

  Logger.log(`Created: ${title} | ${fmtDate_(box.start)} ${fmtTime_(box.start)}–${fmtTime_(box.end)}`);
}

// Warn about missing outline support once per run, not once per event.
let outlineStyleWarned_ = false;

/**
 * Renders an event outline-only (no fill).
 *
 * Google Calendar has no native "outline" event style; the only chips it
 * draws without a fill are invitations the user has not yet responded to.
 * This helper recreates that state on a self-owned event: it patches the
 * event via the Calendar advanced service so the owner appears in the
 * attendee list with responseStatus 'needsAction'. The calendar UI then
 * draws the event as an outlined, unfilled box with the event color used
 * for the border and text.
 *
 * Requires the Calendar advanced service (SETUP step 4: Editor → Services
 * → + → "Google Calendar API", identifier "Calendar"). When the service
 * is not enabled — or the patch fails — the event is left as a normal
 * solid block and a one-time warning is logged.
 *
 * No invitation email is generated: the only attendee is the owner, and
 * the patch does not request notifications.
 */
function applyOutlineStyle_(event) {
  if (typeof Calendar === 'undefined') {
    if (!outlineStyleWarned_) {
      outlineStyleWarned_ = true;
      Logger.log('⚠️ Calendar advanced service not enabled — twilight boxes will render solid. '
        + 'Enable it: Editor → Services → + → "Google Calendar API" (identifier "Calendar").');
    }
    return false;
  }

  try {
    const email = Session.getEffectiveUser().getEmail();
    if (!email) throw new Error('effective user email unavailable');

    // CalendarApp IDs look like "<apiId>@google.com"; the advanced service
    // wants the bare apiId.
    const eventId = event.getId().split('@')[0];

    Calendar.Events.patch(
      { attendees: [{ email, responseStatus: 'needsAction' }] },
      CALENDAR_ID,
      eventId
    );
    return true;
  } catch (e) {
    if (!outlineStyleWarned_) {
      outlineStyleWarned_ = true;
      Logger.log(`⚠️ Outline styling failed (${e.message}) — twilight boxes will render solid.`);
    }
    return false;
  }
}

function createRainCalendarBlock_(block) {
  const calendar = CalendarApp.getCalendarById(CALENDAR_ID);
  const rounded = Math.round(block.peakValue * 10) / 10;

  const title = block.severity === 'Heavy'
    ? `🌧️ Heavy Rain (${rounded} mm/hr)`
    : `🌦️ Rain (${rounded} mm/hr)`;

  const description = [
    `Rain intensity: ${block.severity}`,
    `Peak precipitation: ${rounded} mm/hr`,
    '',
    'Auto-generated by Weather Calendar Blocker.',
    'Rain data: Open-Meteo (GFS). Bring an umbrella!'
  ].join('\n');

  const event = calendar.createEvent(title, block.start, block.end, { description });
  event.setTransparency(CalendarApp.EventTransparency.TRANSPARENT);
  event.setColor(
    block.severity === 'Heavy'
      ? CalendarApp.EventColor.BLUE
      : CalendarApp.EventColor.PALE_BLUE
  );

  try {
    event.setTag('weather-blocker', 'true');
  } catch (e) {}

  Logger.log(`Created: ${title} | ${fmtDate_(block.start)} ${fmtTime_(block.start)}–${fmtTime_(block.end)}`);
}

// ============ HELPERS ============
function fmtTime_(date) {
  return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}

function fmtDate_(date) {
  return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}

// ============ TEST ============
function testSetup() {
  Logger.log('=== Weather Calendar Blocker v10 — All-Tier Daylight Clamp + Twilight Outline Boxes ===');
  Logger.log(`Location: ${LAT}, ${LNG} (${TIMEZONE})`);
  Logger.log(`Forecast window: ${FORECAST_DAYS} days`);
  Logger.log(`UV tiers: very low = daylight & UV < ${UV_VERY_LOW_MAX}, low ${UV_VERY_LOW_MAX}–${UV_LOW_MAX}, medium ${UV_LOW_MAX}–${UV_HIGH_THRESHOLD}, high ≥ ${UV_HIGH_THRESHOLD}`);
  Logger.log(`UV disagreement flag: ≥ ${UV_DISAGREE_THRESHOLD} difference → ⚠️ (gray)`);
  Logger.log(`UV gap bridging: ${UV_GAP_BRIDGE_MS > 0 ? `≤ ${UV_GAP_BRIDGE_MS / 60000} min` : 'disabled'}`);
  Logger.log(`Very Low edge snap: ${UV_EDGE_SNAP_MAX_MS > 0 ? `gaps ≤ ${UV_EDGE_SNAP_MAX_MS / 60000} min snap to sunrise/sunset` : 'disabled (trim only)'}`);
  Logger.log('Daylight clamp: all UV tiers trimmed to exact sunrise/sunset (v10)');
  Logger.log(`Twilight boxes: ${TWILIGHT_MINUTES > 0 ? `${TWILIGHT_MINUTES} min before sunrise / after sunset, outline-only` : 'disabled'}`);
  Logger.log(`Rain threshold: moderate ≥ ${RAIN_MODERATE_MM} mm/hr, heavy ≥ ${RAIN_HEAVY_MM} mm/hr`);

  const { windowStart, windowEnd } = getForecastWindow_();
  Logger.log(`Window: ${fmtDate_(windowStart)} – ${fmtDate_(windowEnd)}`);

  const scriptTz = Session.getScriptTimeZone();
  Logger.log(`Project timezone: ${scriptTz}`);
  if (scriptTz !== TIMEZONE) {
    Logger.log(`⚠️ TIMEZONE MISMATCH: Project (${scriptTz}) vs script (${TIMEZONE}).`);
  }

  if (TWILIGHT_MINUTES > 0) {
    Logger.log(typeof Calendar !== 'undefined'
      ? '✅ Calendar advanced service enabled — twilight boxes render outline-only.'
      : '⚠️ Calendar advanced service NOT enabled — twilight boxes will render solid green. Enable: Editor → Services → + → "Google Calendar API".');
  }
  Logger.log('');

  // ---- Fetch both UV sources ----
  Logger.log('--- UV Sources ---');
  const uvNoaa = fetchUvForecastNoaa_();
  const uvOpenMeteo = fetchUvForecastOpenMeteo_();

  // Show per-day comparison
  if (uvNoaa?.length && uvOpenMeteo?.length) {
    Logger.log('');
    Logger.log('Per-day UV comparison (peak values):');

    const noaaByDay = groupPeakByDay_(uvNoaa, 'uv');
    const omByDay = groupPeakByDay_(uvOpenMeteo, 'uv');

    const allDays = new Set([...Object.keys(noaaByDay), ...Object.keys(omByDay)]);
    for (const day of [...allDays].sort()) {
      const n = noaaByDay[day] ?? '—';
      const o = omByDay[day] ?? '—';
      const diff = (typeof n === 'number' && typeof o === 'number') ? Math.abs(n - o) : null;
      const flag = (diff !== null && diff >= UV_DISAGREE_THRESHOLD) ? ' ⚠️' : '';
      Logger.log(`  ${day}: NOAA=${n}  Open-Meteo=${o}  (diff: ${diff !== null ? diff.toFixed(1) : '?'})${flag}`);
    }
  }

  // ---- Sun times ----
  Logger.log('');
  Logger.log('--- Sun Times: Open-Meteo ---');
  const sunTimes = fetchSunTimes_();
  if (sunTimes?.length) {
    for (const d of sunTimes) {
      Logger.log(`  ${fmtDate_(d.sunrise)}: sunrise ${fmtTime_(d.sunrise)}, solar noon ${fmtTime_(d.solarNoon)}, sunset ${fmtTime_(d.sunset)}`);
    }
  } else {
    Logger.log('❌ Sun times unavailable — Very Low tier will use UV > 0 fallback,');
    Logger.log('   Low/Med/High will stay slot-aligned, and twilight boxes will be skipped.');
  }

  // Merge and build blocks
  const uvMerged = mergeUvSources_(uvNoaa, uvOpenMeteo);

  if (uvMerged?.length) {
    const uvSlots = interpolateUvToSlots_(uvMerged);

    const medHighRaw = buildMedHighUvBlocks_(uvSlots);
    const medHighBridged = bridgeGaps_(medHighRaw, UV_GAP_BRIDGE_MS);
    const bridged = medHighRaw.length - medHighBridged.length;
    const medHighClamped = trimToDaylight_(medHighBridged, sunTimes, uvSlots, 'Med/High UV');
    const medHighBlocks = splitAtSolarNoon_(medHighClamped, sunTimes, uvSlots);
    const lowClamped = trimToDaylight_(buildLowUvBlocks_(uvSlots), sunTimes, uvSlots, 'Low UV');
    const lowBlocks = splitAtSolarNoon_(lowClamped, sunTimes, uvSlots);
    const veryLowBlocks = buildVeryLowUvBlocks_(uvSlots, sunTimes);

    const allUvBlocks = [...medHighBlocks, ...lowBlocks, ...veryLowBlocks];
    const flagged = allUvBlocks.filter(b => b.lowConfidence).length;
    const total = allUvBlocks.length;

    Logger.log('');
    Logger.log(`Would create ${total} UV block(s)`
      + ` (${medHighBlocks.length} med/high, ${lowBlocks.length} low, ${veryLowBlocks.length} very low)`
      + (bridged > 0 ? `, ${bridged} gap(s) bridged` : '')
      + (flagged > 0 ? `, ${flagged} flagged ⚠️` : '')
      + ':');

    for (const b of medHighBlocks) {
      const flag = b.lowConfidence ? ' ⚠️' : '';
      Logger.log(`  UV ${b.severity}${flag}: ${fmtDate_(b.start)} ${fmtTime_(b.start)}–${fmtTime_(b.end)} (peak ${Math.round(b.peakValue * 10) / 10})`);
    }
    for (const b of lowBlocks) {
      const flag = b.lowConfidence ? ' ⚠️' : '';
      Logger.log(`  UV Low${flag}: ${fmtDate_(b.start)} ${fmtTime_(b.start)}–${fmtTime_(b.end)} (peak ${Math.round(b.peakValue * 10) / 10})`);
    }
    for (const b of veryLowBlocks) {
      const flag = b.lowConfidence ? ' ⚠️' : '';
      Logger.log(`  UV Very Low${flag}: ${fmtDate_(b.start)} ${fmtTime_(b.start)}–${fmtTime_(b.end)} (peak ${Math.round(b.peakValue * 10) / 10})`);
    }
  }

  // ---- Twilight boxes ----
  if (TWILIGHT_MINUTES > 0 && sunTimes?.length) {
    const twilightBoxes = buildTwilightBoxes_(sunTimes);
    Logger.log('');
    Logger.log(`Would create ${twilightBoxes.length} twilight box(es) (outline-only):`);
    for (const b of twilightBoxes) {
      Logger.log(`  ${b.kind === 'Dawn' ? '🌅 Dawn' : '🌇 Dusk'}: ${fmtDate_(b.start)} ${fmtTime_(b.start)}–${fmtTime_(b.end)}`);
    }
  }

  Logger.log('');

  // ---- Rain ----
  Logger.log('--- Rain Source: Open-Meteo (GFS) ---');
  const rainHourly = fetchRainForecast_();

  if (rainHourly?.length) {
    const rainSlots = flatExpandToSlots_(rainHourly);
    const peakRain = rainHourly.reduce((max, h) => h.precipitation > max.precipitation ? h : max, rainHourly[0]);
    if (peakRain.precipitation > 0) {
      Logger.log(`Peak rain: ${Math.round(peakRain.precipitation * 10) / 10} mm/hr at ${fmtTime_(peakRain.time)} on ${fmtDate_(peakRain.time)}`);
    } else {
      Logger.log('No precipitation in forecast window.');
    }

    const rainBlocks = buildRainBlocks_(rainSlots);
    Logger.log(`Would create ${rainBlocks.length} rain block(s):`);
    for (const b of rainBlocks) {
      Logger.log(`  Rain ${b.severity}: ${fmtDate_(b.start)} ${fmtTime_(b.start)}–${fmtTime_(b.end)} (peak ${Math.round(b.peakValue * 10) / 10} mm/hr)`);
    }
  } else {
    Logger.log('❌ Rain API call failed.');
  }

  // ---- Trigger status ----
  Logger.log('');
  const triggers = ScriptApp.getProjectTriggers()
    .filter(t => [TRIGGER_FUNCTION, 'createUvCalendarBlocks'].includes(t.getHandlerFunction()));
  if (triggers.length >= 2) {
    Logger.log(`✅ Twice-daily triggers active (${triggers.length} trigger(s) found).`);
  } else if (triggers.length === 1) {
    Logger.log(`⚠️ Only 1 trigger found — run "installTriggers" to set up twice-daily runs.`);
  } else {
    Logger.log('⚠️ No triggers found. Run "installTriggers" to set up twice-daily runs.');
  }
}

/**
 * Groups hourly data by date and returns the peak value per day.
 */
function groupPeakByDay_(hourlyData, field) {
  const byDay = {};
  for (const e of hourlyData) {
    const dayKey = fmtDate_(e.time);
    const val = e[field] ?? 0;
    if (!(dayKey in byDay) || val > byDay[dayKey]) {
      byDay[dayKey] = Math.round(val * 10) / 10;
    }
  }
  return byDay;
}
