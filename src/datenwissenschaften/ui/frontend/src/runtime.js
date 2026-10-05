const MINUTE_MS = 60000
const MINUTES_PER_HOUR = 60
const HOURS_PER_DAY = 24

export const elapsed = (startedAt, now) => {
  const minutes = Math.max(0, Math.floor((now.getTime() - new Date(startedAt).getTime()) / MINUTE_MS))
  const hours = Math.floor(minutes / MINUTES_PER_HOUR)
  const days = Math.floor(hours / HOURS_PER_DAY)
  if (days) return `${days}d ${hours % HOURS_PER_DAY}h`
  if (hours) return `${hours}h ${minutes % MINUTES_PER_HOUR}m`
  return `${minutes}m`
}

export const minutesLeft = (until, now) => Math.max(1, Math.ceil((new Date(until).getTime() - now.getTime()) / MINUTE_MS))
