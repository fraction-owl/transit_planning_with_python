/**
 * Weather Calendar Blocker v9.2 — All-Tier Daylight Clamp + Twilight Outline Buffers
 *
 * UV tiers (traffic-light + very low):
 *   🌿 Very Low  (daylight, UV < 1.0) → Green  — shoulder/safe hours (sunrise–sunset)
 *   🌤️ Low       (UV 1.0 – 2.9)   → Yellow — minimal risk
 *   ⛅ Medium    (UV 3.0 – 5.9)   → Orange — apply sunscreen
 *   ☀️ High      (UV 6+)           → Red    — minimize exposure
 *   ⚠️ any tier  → Gray when NOAA and Open-Meteo disagree by ≥ UV_DISAGREE_THRESHOLD
 *   (Gray replaces the old yellow confidence override, freeing yellow for the Low tier.)
 *   🌘 Twilight  (30 min before sunrise / after sunset) → outline-only, no fill —
 *   trace amounts of ambient UV still reach ground level during civil twilight.
 *
 * UV data:   CurrentUVIndex.com (NOAA) + Open-Meteo (GFS-derived)
 *            When both sources agree → confident block
 *            When they disagree → block with ⚠️ flag (gray override)
 * Rain data: Open-Meteo (GFS model, no API key)
 * Sun times: Open-Meteo daily sunrise/sunset — clamps ALL UV tiers to daylight
 *            and positions the twilight buffers. Sunrise/sunset move through
 *            the year; fetching them per-day from the API keeps the clamp
 *            accurate in every season. If the sun-times fetch fails, Very Low
 *            falls back to the old UV > 0 heuristic for that run, the other
 *            tiers go unclamped, and twilight buffers are skipped.
 *
 * ROLLING 5-DAY WINDOW with TWICE-DAILY updates:
 *   Runs at ~6:15 AM and ~12:15 PM. Each run deletes all weather blocks
 *   from TODAY forward and recreates with the freshest forecast data.
 *   Past blocks are preserved as historical record.
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
 * v9.2 — All-tier daylight clamp + twilight outline buffers:
 *   (1) EVERY UV tier is now clamped to the day's exact sunrise/sunset.
 *       v9.1 clamped only Very Low, so a Low/Medium/High block whose
 *       edge slot straddled a sun time (interpolation can put UV ≥ 1 on
 *       a slot that starts before sunrise or ends after sunset) bled
 *       outside daylight. Low/Med/High get a trim-only clamp (no snap —
 *       snapping is a Very Low-only recovery for missing morning data).
 *   (2) New 🌘 Twilight buffer blocks: TWILIGHT_BUFFER_MINUTES (30) before
 *       sunrise and after sunset, rendered with NO fill and only an
 *       outline. Google Calendar has no native outline style; the blocks
 *       are created via the Calendar advanced service with the owner as
 *       an unresponded attendee — Calendar renders unanswered invitations
 *       hollow (white fill, colored outline). If the advanced service is
 *       not enabled, they fall back to filled graphite CalendarApp events.
 *       Twilight buffers only need sun times, so they are created even on
 *       runs where both UV sources fail.
 *
 * The UV > 0 fallback path (sun-times fetch failed) is slot-aligned only
 * — no trim, snap, or split.
 *
 * REQUIRES: V8 runtime (set in appsscript.json or Project Settings → Runtime)
 *
 * SETUP:
 * 1. Go to https://script.google.com → New Project
 * 2. Delete the placeholder code and paste this entire file
 * 3. Also create/edit appsscript.json (View → Show manifest file) — enable the
 *    Calendar advanced service so twilight buffers render outline-only:
 *      "dependencies": { "enabledAdvancedServices": [
 *        { "userSymbol": "Calendar", "serviceId": "calendar", "version": "v3" } ] }
 *    (Or: Editor sidebar → Services → + → Calendar API → Add.)
 * 4. Adjust LAT, LNG, and TIMEZONE below if needed (defaults to DC area)
 * 5. Save (Ctrl+S)
 * 6. Select "testSetup" → Run (▶️) — first run asks for permissions
 * 7. Select "installTriggers" → Run once to set up twice-daily auto-runs
 * 8. Optionally run "createWeatherCalendarBlocks" to test a real calendar write
 *
 * RUN DROPDOWN: Only the four entry points appear in the editor's Run menu —
 *   createWeatherCalendarBlocks, installTriggers, removeAllTriggers, testSetup.
 * Every other function name ends in "_", which Apps Script treats as private
 * (hidden from the dropdown and not directly runnable). Once installTriggers
 * has been run, no manual runs are needed — the script maintains itself.
 *
 * UPGRADING from v7 or earlier: paste this file over the old code, save, and
 * run "installTriggers" once. Triggers pointing at removed legacy names
 * (e.g. createUvCalendarBlocks) are cleaned up automatically; leaving them
 * in place would cause failed-trigger emails since those functions no
 * longer exist.
 *
 * UV data credit: https://currentuvindex.com (CC BY 4.0)
 */

// ============ CONFIGURATION ============

const LAT = 38.9072;                   // Your latitude (default: Washington DC area)
const LNG = -77.0369;                  // Your longitude
const TIMEZONE = 'America/New_York';   // Your timezone
const FORECAST_DAYS = 5;               // Rolling forecast window
const UV_VERY_LOW_MAX = 1.0;           // Daylight, UV < 1.0 → Very Low (green)
const UV_LOW_MAX = 3.0;                // 1.0 ≤ UV < 3.0 → Low (yellow)
const UV_HIGH_THRESHOLD = 6.0;         // UV ≥ 6.0 → High (red); 3.0–5.9 → Medium (orange)
const UV_GAP_BRIDGE_MS = 0;            // Bridge Medium/High gaps ≤ this ms; 0 = disabled.
                                       // Re-enable (e.g. 60 * 60 * 1000) with caution:
                                       // bridging across tier boundaries causes overlap.
const UV_DISAGREE_THRESHOLD = 1.5;     // Sources differ by 1.5+ = flag ⚠️ (gray)
const UV_EDGE_SNAP_MAX_MS = 60 * 60 * 1000; // Snap the day's first/last Very Low block
                                       // edge to exact sunrise/sunset if the gap is ≤ this
                                       // AND the gap has no elevated-UV data (i.e. the gap
                                       // is missing data, e.g. the 6:15 AM run whose NOAA
                                       // "now" entry starts after sunrise). 0 = trim-only.
const TWILIGHT_BUFFER_MINUTES = 30;    // Outline-only buffer before sunrise / after sunset
                                       // marking trace ground-level UV during civil twilight.
const RAIN_MODERATE_MM = 2.5;          // 2.5+ mm/hr = moderate rain
const RAIN_HEAVY_MM = 7.6;             // 7.6+ mm/hr = heavy rain
const CALENDAR_ID = 'primary';         // Use your primary Google Calendar
const TRIGGER_FUNCTION = 'createWeatherCalendarBlocks';
const SLOT_MINUTES = 30;               // Calendar block granularity

// ============ MAIN FUNCTION ============

function createWeatherCalendarBlocks() {
  try {
    // Fetch UV from both sources
    const uvNoaa = fetchUvForecastNoaa_();
    const uvOpenMeteo = fetchUvForecastOpenMeteo_();

    // Fetch rain from Open-Meteo
    const rainHourly = fetchRainForecast_();

    // Fetch sunrise/sunset — clamps all UV tiers and positions twilight buffers
    const sunTimes = fetchSunTimes_();

    if (!uvNoaa && !uvOpenMeteo && !rainHourly && !sunTimes) {
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
      const medHighBlocks = splitAtSolarNoon_(
        trimToDaylight_(medHighBridged, sunTimes), sunTimes, uvSlots);

      // Low blocks (UV_VERY_LOW_MAX ≤ UV < UV_LOW_MAX) — not bridged
      const lowBlocks = splitAtSolarNoon_(
        trimToDaylight_(buildLowUvBlocks_(uvSlots), sunTimes), sunTimes, uvSlots);

      // Very Low blocks (daylight, UV < UV_VERY_LOW_MAX) — not bridged
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

    // Twilight buffer blocks — outline-only, before sunrise / after sunset.
    // Only need sun times, so they're created even when both UV sources fail.
    let twilightCount = 0;
    if (sunTimes?.length) {
      const twilightBlocks = buildTwilightBlocks_(sunTimes);
      for (const block of twilightBlocks) createTwilightCalendarBlock_(block);
      twilightCount = twilightBlocks.length;
      Logger.log(`Created ${twilightCount} twilight buffer block(s).`);
    } else {
      Logger.log('Sun times unavailable — skipping twilight buffer blocks.');
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
 *   - NOAA is primary (more direct UV modeling from NOAA data)
 *   - Open-Meteo is comparison source (GFS-derived approximation)
 *   - For each hour, we use NOAA's value but flag low confidence
 *     when the two sources disagree by ≥ UV_DISAGREE_THRESHOLD
 *   - If NOAA fails, fall back to Open-Meteo entirely (all flagged)
 *   - If Open-Meteo fails, use NOAA without confidence info
 *
 * Returns array of { time, uv, lowConfidence, noaaUv, openMeteoUv }
 */
function mergeUvSources_(noaaData, openMeteoData) {
  // Both failed
  if (!noaaData?.length && !openMeteoData?.length) return null;

  // Only Open-Meteo available — use it but flag everything
  if (!noaaData?.length && openMeteoData?.length) {
    Logger.log('⚠️  NOAA UV unavailable — using Open-Meteo only (all blocks flagged).');
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
 * Used to clamp all UV tiers to daylight and position twilight buffers.
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
 * daylight period. Testing the slot *interval* (rather than its start
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
    : entry => entry.uv > 0 && entry.uv < UV_VERY_LOW_MAX;  // fallback

  if (!sunTimes?.length) {
    Logger.log('⚠️  Sun times unavailable — Very Low tier using UV > 0 fallback.');
  }

  const blocks = buildConsecutiveBlocks_(slotData, filterFn, () => 'Very Low', 'uv');
  if (!sunTimes?.length) return blocks;

  const clamped = clampToDaylight_(blocks, sunTimes, slotData);
  return splitAtSolarNoon_(clamped, sunTimes, slotData);
}

/**
 * Trim-only daylight clamp for Low/Medium/High blocks. Interpolation can
 * put UV ≥ UV_VERY_LOW_MAX on an edge slot that starts before sunrise or
 * ends after sunset; clamping cuts those edges back to the exact sun time
 * so no elevated-UV block bleeds outside daylight. Unlike the Very Low
 * path there is no snap — a Low/Med/High block that starts after sunrise
 * genuinely starts there (the shoulder before it belongs to Very Low).
 *
 * A block that overlaps no sun-times day (data beyond the sun-times
 * window) passes through unchanged rather than being dropped. A block
 * fully outside daylight within a covered day is dropped. No-op when
 * sun times are unavailable.
 */
function trimToDaylight_(blocks, sunTimes) {
  if (!sunTimes?.length) return blocks;

  const out = [];

  for (const block of blocks) {
    const day = sunTimes.find(d => block.start < d.sunset && block.end > d.sunrise);

    if (!day) {
      out.push(block);
      continue;
    }

    const start = block.start < day.sunrise ? day.sunrise : block.start;
    const end = block.end > day.sunset ? day.sunset : block.end;
    if (start < end) out.push({ ...block, start, end });
  }

  return out;
}

/**
 * Clamps block edges to exact sunrise/sunset, working one daylight period
 * at a time. Two operations per day:
 *
 *   TRIM: edge slots straddling sunrise/sunset are cut back to the exact
 *   sun time (unchanged from v8). A block with no daylight overlap is
 *   dropped (shouldn't happen — every slot passed overlapsDaylight_).
 *
 *   SNAP: the day's first block is extended back to start at sunrise, and
 *   the day's last block extended forward to end at sunset, IF the gap is
 *   ≤ UV_EDGE_SNAP_MAX_MS and contains no elevated-UV slot. Trim alone
 *   could only shrink blocks, so a run whose data began after sunrise
 *   (the 6:15 AM trigger) produced an AM block starting mid-morning. The
 *   elevated-UV guard keeps midday dip blocks in place: a gap between
 *   sunrise and a 10 AM dip block is real Low/Med UV, not missing data,
 *   and must not be painted green.
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
 * Builds the twilight buffer blocks: one ending exactly at sunrise and
 * one starting exactly at sunset, each TWILIGHT_BUFFER_MINUTES long.
 * They mark the civil-twilight window when a trace of ambient UV still
 * reaches ground level. Pure sun-time geometry — no UV data involved —
 * so they exist even on runs where both UV sources fail.
 */
function buildTwilightBlocks_(sunTimes) {
  if (!sunTimes?.length) return [];

  const bufferMs = TWILIGHT_BUFFER_MINUTES * 60 * 1000;
  const blocks = [];

  for (const day of sunTimes) {
    blocks.push({
      phase: 'dawn',
      start: new Date(day.sunrise.getTime() - bufferMs),
      end: day.sunrise
    });
    blocks.push({
      phase: 'dusk',
      start: day.sunset,
      end: new Date(day.sunset.getTime() + bufferMs)
    });
  }

  return blocks;
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
      // Twilight blocks are matched by title only: created via the advanced
      // Calendar API, they carry an extendedProperty rather than a
      // CalendarApp tag, so getTag above doesn't see them.
      isWeatherBlock =
        /^(☀️ UV|⚠️☀️ UV|⛅ UV|⚠️⛅ UV|🌤️ UV|⚠️🌤️ UV|🌿 UV|⚠️🌿 UV|🌘 UV Twilight|🌧️ Heavy Rain|🌦️ Rain)/.test(title) ||
        /UV (High|Medium|Low|Very Low|Twilight)/.test(title) ||
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
  } else { // Very Low — daylight shoulder hours
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
  } else { // Very Low
    event.setColor(CalendarApp.EventColor.GREEN);
  }

  try {
    event.setTag('weather-blocker', 'true');
  } catch (e) {}

  Logger.log(`Created: ${title} | ${fmtDate_(block.start)} ${fmtTime_(block.start)}–${fmtTime_(block.end)}`);
}

/**
 * Creates a twilight buffer block, rendered outline-only (no fill).
 *
 * Google Calendar has no fill-style API. The one style it renders hollow
 * — white fill, colored outline — is an invitation you haven't responded
 * to. So the block is inserted via the Calendar ADVANCED service with the
 * calendar owner as an attendee whose responseStatus stays 'needsAction';
 * Calendar then draws it as an outline. The event is still deleted
 * normally by removeWeatherBlocksFromTodayForward_ (it matches by title).
 * Graphite (colorId 8) keeps the outline neutral so the hollow box reads
 * as "trace UV", below the green Very Low tier.
 *
 * If the advanced Calendar service is not enabled in the manifest, falls
 * back to a normal (filled) graphite CalendarApp event — same time range,
 * just without the hollow rendering.
 */
function createTwilightCalendarBlock_(block) {
  const title = block.phase === 'dawn'
    ? '🌘 UV Twilight (pre-sunrise)'
    : '🌘 UV Twilight (post-sunset)';

  const description = [
    `Civil twilight buffer: ${TWILIGHT_BUFFER_MINUTES} min ${block.phase === 'dawn' ? 'before sunrise' : 'after sunset'}.`,
    'Small amounts of ambient UV still reach ground level during this window.',
    '',
    'Auto-generated by Weather Calendar Blocker.',
    'Sun times: Open-Meteo.'
  ].join('\n');

  if (typeof Calendar !== 'undefined' && Calendar?.Events?.insert) {
    // Primary calendar ID is the owner's email; fall back to it if the
    // session email is unavailable (some trigger contexts).
    const selfEmail = Session.getEffectiveUser().getEmail()
      || CalendarApp.getDefaultCalendar().getId();

    Calendar.Events.insert({
      summary: title,
      description,
      start: { dateTime: block.start.toISOString() },
      end: { dateTime: block.end.toISOString() },
      transparency: 'transparent',
      colorId: '8', // graphite outline
      attendees: [{ email: selfEmail, responseStatus: 'needsAction' }],
      extendedProperties: { private: { 'weather-blocker': 'true' } }
    }, CALENDAR_ID, { sendUpdates: 'none' });

    Logger.log(`Created (outline): ${title} | ${fmtDate_(block.start)} ${fmtTime_(block.start)}–${fmtTime_(block.end)}`);
    return;
  }

  Logger.log('⚠️  Calendar advanced service not enabled — twilight block will render filled, not outlined.');

  const calendar = CalendarApp.getCalendarById(CALENDAR_ID);
  const event = calendar.createEvent(title, block.start, block.end, { description });
  event.setTransparency(CalendarApp.EventTransparency.TRANSPARENT);
  event.setColor(CalendarApp.EventColor.GRAY);

  try {
    event.setTag('weather-blocker', 'true');
  } catch (e) {}

  Logger.log(`Created (filled fallback): ${title} | ${fmtDate_(block.start)} ${fmtTime_(block.start)}–${fmtTime_(block.end)}`);
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
  Logger.log('=== Weather Calendar Blocker v9.2 — All-Tier Daylight Clamp + Twilight Outline Buffers ===');
  Logger.log(`Location: ${LAT}, ${LNG} (${TIMEZONE})`);
  Logger.log(`Forecast window: ${FORECAST_DAYS} days`);
  Logger.log(`UV tiers: very low = daylight & UV < ${UV_VERY_LOW_MAX}, low ${UV_VERY_LOW_MAX}–${UV_LOW_MAX}, medium ${UV_LOW_MAX}–${UV_HIGH_THRESHOLD}, high ≥ ${UV_HIGH_THRESHOLD}`);
  Logger.log(`UV disagreement flag: ≥ ${UV_DISAGREE_THRESHOLD} difference → ⚠️ (gray)`);
  Logger.log(`UV gap bridging: ${UV_GAP_BRIDGE_MS > 0 ? `≤ ${UV_GAP_BRIDGE_MS / 60000} min` : 'disabled'}`);
  Logger.log(`Very Low edge snap: ${UV_EDGE_SNAP_MAX_MS > 0 ? `gaps ≤ ${UV_EDGE_SNAP_MAX_MS / 60000} min snap to sunrise/sunset` : 'disabled (trim only)'}`);
  Logger.log(`Daylight clamp: all UV tiers trimmed to exact sunrise/sunset`);
  Logger.log(`Twilight buffers: ${TWILIGHT_BUFFER_MINUTES} min before sunrise / after sunset (outline-only)`);
  Logger.log(`Rain threshold: moderate ≥ ${RAIN_MODERATE_MM} mm/hr, heavy ≥ ${RAIN_HEAVY_MM} mm/hr`);

  const { windowStart, windowEnd } = getForecastWindow_();
  Logger.log(`Window: ${fmtDate_(windowStart)} – ${fmtDate_(windowEnd)}`);

  const scriptTz = Session.getScriptTimeZone();
  Logger.log(`Project timezone: ${scriptTz}`);
  if (scriptTz !== TIMEZONE) {
    Logger.log(`⚠️  TIMEZONE MISMATCH: Project (${scriptTz}) vs script (${TIMEZONE}).`);
  }

  if (typeof Calendar !== 'undefined' && Calendar?.Events) {
    Logger.log('✅ Calendar advanced service enabled — twilight blocks will render outline-only.');
  } else {
    Logger.log('⚠️  Calendar advanced service NOT enabled — twilight blocks will fall back to filled graphite. See SETUP step 3.');
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

    const twilightPreview = buildTwilightBlocks_(sunTimes);
    Logger.log('');
    Logger.log(`Would create ${twilightPreview.length} twilight outline block(s) (${TWILIGHT_BUFFER_MINUTES} min each):`);
    for (const b of twilightPreview) {
      Logger.log(`  UV Twilight (${b.phase}): ${fmtDate_(b.start)} ${fmtTime_(b.start)}–${fmtTime_(b.end)}`);
    }
  } else {
    Logger.log('❌ Sun times unavailable — Very Low tier will use UV > 0 fallback; no daylight clamp or twilight blocks.');
  }

  // Merge and build blocks
  const uvMerged = mergeUvSources_(uvNoaa, uvOpenMeteo);

  if (uvMerged?.length) {
    const uvSlots = interpolateUvToSlots_(uvMerged);

    const medHighRaw = buildMedHighUvBlocks_(uvSlots);
    const medHighBridged = bridgeGaps_(medHighRaw, UV_GAP_BRIDGE_MS);
    const bridged = medHighRaw.length - medHighBridged.length;
    const medHighBlocks = splitAtSolarNoon_(
      trimToDaylight_(medHighBridged, sunTimes), sunTimes, uvSlots);
    const lowBlocks = splitAtSolarNoon_(
      trimToDaylight_(buildLowUvBlocks_(uvSlots), sunTimes), sunTimes, uvSlots);
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

  Logger.log('');

  // ---- Rain ----
  Logger.log('--- Rain Source: Open-Meteo (GFS) ---');
  const rainHourly = fetchRainForecast_();

  if (rainHourly?.length) {
    const rainSlots = flatExpandToSlots_(rainHourly);
    const peakRain = rainHourly.reduce((max, h) =>
      h.precipitation > max.precipitation ? h : max, rainHourly[0]);
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
    Logger.log(`⚠️  Only 1 trigger found — run "installTriggers" to set up twice-daily runs.`);
  } else {
    Logger.log('⚠️  No triggers found. Run "installTriggers" to set up twice-daily runs.');
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
