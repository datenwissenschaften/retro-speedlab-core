export const STREAM_TIME_ZONE = 'Europe/Berlin'
const RESTART_HOUR = 4
const RESTART_MINUTE = 0
const CHECK_INTERVAL_MS = 10000
const FULL_CONTROL_LEVEL = 5

const clock = new Intl.DateTimeFormat('en-CA', {
  timeZone: STREAM_TIME_ZONE,
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hourCycle: 'h23',
})

const localTime = () => {
  const parts = Object.fromEntries(clock.formatToParts(new Date()).map(({ type, value }) => [type, value]))
  return { day: `${parts.year}-${parts.month}-${parts.day}`, hour: Number(parts.hour), minute: Number(parts.minute) }
}

export const createObsControl = () => {
  const obs = window.obsstudio
  let restartedOn = null
  let timer

  const restartDue = () => {
    const { day, hour, minute } = localTime()
    if (hour !== RESTART_HOUR || minute !== RESTART_MINUTE || day === restartedOn) return false
    restartedOn = day
    return true
  }
  const keepStreaming = () => {
    const restart = restartDue()
    obs.getStatus(status => {
      if (!status.streaming) obs.startStreaming()
      else if (restart) obs.stopStreaming()
    })
  }
  const schedule = level => {
    if (level < FULL_CONTROL_LEVEL) {
      console.error('Keeping the OBS stream alive needs the browser source permission "Full access to OBS"')
      return
    }
    keepStreaming()
    timer = window.setInterval(keepStreaming, CHECK_INTERVAL_MS)
  }

  return {
    start() { if (obs) obs.getControlLevel(schedule) },
    stop() { window.clearInterval(timer) },
  }
}
