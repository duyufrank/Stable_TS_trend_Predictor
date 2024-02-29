# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 10:58:13 2021

@author: Yu Du
"""
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
def get_indi(df, h=6, c=15, showPlots=False):
    def plotIndicator(df1, df2, showPlots=True, savePlots=None, namePrefix='', nameSuffix=''):
        mpl.style.use('classic')  # old matplotlib visualization style
        if not (showPlots or savePlots):
            print(
                'Warning: There is no point in running compareTwoIndicators() without either showing or saving the plots!')
        if df1.columns[0] == df2.columns[0]:
            plotMe = pd.concat([df1, df2.rename(columns={df2.columns[0]: 'ind'})], axis=1)
        else:
            plotMe = pd.concat([df1, df2], axis=1)
        plotMe.dropna(how='all', inplace=True)
        plotMe['time'] = range(len(plotMe))
        col1 = plotMe.columns[0]
        col2 = plotMe.columns[1]
        if (len(plotMe) < 12 * 25):
            data_values = [row['time'] for index, row in plotMe.iterrows() if index.month == 1]
            data_labels = [index.strftime('%Y-%m') for index, row in plotMe.iterrows() if index.month == 1]
        else:
            data_values = [row['time'] for index, row in plotMe.iterrows() if
                           (index.month == 1 & (index.year % 5 == 0))]
            data_labels = [index.strftime('%Y-%m') for index, row in plotMe.iterrows() if
                           (index.month == 1 & (index.year % 5 == 0))]
        plt.ioff()  # Turn interactive plotting off
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 2.5))
        fig.patch.set_facecolor('white')
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')
        plt.xticks(data_values, data_labels, fontsize=10, rotation=60)
        plt.yticks(fontsize=10)
        ax.plot(plotMe[-plotMe[col1].isnull()]['time'], plotMe[-plotMe[col1].isnull()][col1], linestyle='-',
                color='gray')
        for i, row in plotMe.iterrows():
            if row[col2] == 1:
                plt.axvline(x=row['time'], linestyle='--', color='black')
            elif row[col2] == -1:
                plt.axvline(x=row['time'], linestyle='-.', color='black')
        ax.margins(x=0.0, y=0.1)
        fig.tight_layout()
        if savePlots:
            fig.savefig(os.path.join(savePlots, namePrefix + str(col1) + nameSuffix), dpi=300)
        if showPlots:
            plt.show()
        plt.close(fig)

    def pipelineOneColumnTPDetection(col, printDetails=True, showPlots=True, savePlots=None, saveLogs=None, createInverse=False, h=5, c=15):

        def getLocalExtremes(df, showPlots=True, savePlots=None, nameSuffix=''):
            dataShifted = pd.DataFrame(index=df.index)
            for i in range(-5, 5):
                dataShifted = pd.concat([dataShifted, df.shift(i).rename(columns={df.columns[0]: 'shift_' + str(i)})],
                                        axis=1)
            dataInd = pd.DataFrame(0, index=df.index, columns=df.columns)
            dataInd[dataShifted['shift_0'] >= dataShifted.drop('shift_0', axis=1).max(axis=1)] = 1
            dataInd[dataShifted['shift_0'] <= dataShifted.drop('shift_0', axis=1).min(axis=1)] = -1
            dataInd[:5] = 0
            dataInd[-5:] = 0
            if showPlots or savePlots:
                plotIndicator(df, dataInd, showPlots=showPlots, savePlots=savePlots, nameSuffix=nameSuffix)
            return (dataInd)

        def checkAlterations(df, indicator, keepFirst=False, printDetails=True, showPlots=True, savePlots=None, nameSuffix='', saveLogs=None):
            dataInd = indicator.copy()
            checkAlt = dataInd.cumsum()
            if printDetails:
                print('\nChecking extremes at %s for alterations:' % (dataInd.columns[0]))
            if saveLogs:
                saveLogs.write('\nChecking extremes at %s for alterations:' % (dataInd.columns[0]))
            if ((checkAlt.max() - checkAlt.min())[0] > 1):  # are there any non alterating turning points?
                lastExt = 0
                lastDate = df.index[0]
                thisDate = dataInd[dataInd != 0].first_valid_index()
                while thisDate:
                    thisExt = dataInd.loc[thisDate][0]
                    if thisExt == lastExt:  # both local extremes of the same type?
                        if (not (keepFirst) and ((thisExt * df.loc[thisDate])[0] > (lastExt * df.loc[lastDate])[
                            0])):  # keep the higher one (or the earlier one when they equal)
                            if printDetails:
                                print('Deleting extreme (%d) at %s' % (lastExt, str(lastDate)))
                            if saveLogs:
                                saveLogs.write('\nDeleting extreme (%d) at %s' % (lastExt, str(lastDate)))
                            dataInd.loc[lastDate] = 0
                            lastExt = thisExt
                            lastDate = thisDate
                        else:
                            if printDetails:
                                print('Deleting extreme (%d) at %s' % (thisExt, str(thisDate)))
                            if saveLogs:
                                saveLogs.write('\nDeleting extreme (%d) at %s' % (thisExt, str(thisDate)))
                            dataInd.loc[thisDate] = 0
                    else:
                        lastExt = thisExt
                        lastDate = thisDate
                    try:
                        thisDate = dataInd[thisDate:][1:][dataInd != 0].first_valid_index()
                    except IndexError:
                        break
            if showPlots or savePlots:
                plotIndicator(df, dataInd, showPlots=showPlots, savePlots=savePlots, nameSuffix=nameSuffix)
            if saveLogs:
                saveLogs.flush()
            return (dataInd)

        def checkNeighbourhood(df, indicator, printDetails=True, showPlots=True, savePlots=None, nameSuffix='', saveLogs=None):
            dataInd = indicator.copy()
            if printDetails:
                print('\nChecking extremes at %s for higher/lower neighbours:' % (dataInd.columns[0]))
            if saveLogs:
                saveLogs.write('\nChecking extremes at %s for higher/lower neighbours:' % (dataInd.columns[0]))
            lastDate = df.index[0]
            maxDate = dataInd.index[-1]
            if (dataInd[dataInd != 0].first_valid_index() != None):
                thisDate = dataInd[dataInd != 0].first_valid_index()
            else:
                thisDate = maxDate
            while thisDate < maxDate:
                thisExt = dataInd.loc[thisDate][0]
                try:
                    nextDate = dataInd[thisDate:][1:][dataInd != 0].first_valid_index()
                except IndexError:  # previous versions of pandas throw exception
                    nextDate = maxDate
                if nextDate == None:  # newer versions of pandas returns None
                    nextDate = maxDate
                if ((thisExt * df.loc[lastDate:nextDate]).max()[0] > (thisExt * df.loc[thisDate])[
                    0]):  # is there higher/lower point then this max/min?
                    if printDetails:
                        print('Deleting extreme (%d) at %s' % (thisExt, str(thisDate)))
                    if saveLogs:
                        saveLogs.write('\nDeleting extreme (%d) at %s' % (thisExt, str(thisDate)))
                    dataInd.loc[thisDate] = 0
                else:
                    lastDate = thisDate
                thisDate = nextDate
            if showPlots or savePlots:
                plotIndicator(df, dataInd, showPlots=showPlots, savePlots=savePlots, nameSuffix=nameSuffix)
            if saveLogs:
                saveLogs.flush()
            return (dataInd)

        def checkCycleLength(df, indicator, cycleLength=15, printDetails=True, showPlots=True, savePlots=None, nameSuffix='', saveLogs=None):
            dataInd = indicator.copy()
            if printDetails:
                print('\nChecking extremes at %s for cycle length:' % (dataInd.columns[0]))
            if saveLogs:
                saveLogs.write('\nChecking extremes at %s for cycle length:' % (dataInd.columns[0]))
            for thisExt in [-1, 1]:
                if (dataInd[dataInd == thisExt].notnull().sum()[0] > 1):  # more than 1 cycle?
                    lastDate = dataInd[dataInd == thisExt].first_valid_index()
                    thisDate = dataInd[dataInd == thisExt][lastDate:][1:].first_valid_index()
                    while thisDate:
                        realLength = dataInd[lastDate:thisDate].shape[0]
                        if (realLength <= (cycleLength + 1)):  # too short to be a cycle?
                            lastExt = thisExt  # just to be very clear in the next lines
                            if ((thisExt * df.loc[thisDate])[0] > (lastExt * df.loc[lastDate])[0]):  # keep the higher one (or the earlier one when they equal)
                                if printDetails:
                                    print('Deleting extreme (%d) at %s' % (lastExt, str(lastDate)))
                                if saveLogs:
                                    saveLogs.write('\nDeleting extreme (%d) at %s' % (lastExt, str(lastDate)))
                                dataInd.loc[lastDate] = 0
                                lastDate = thisDate
                            else:
                                if printDetails:
                                    print('Deleting extreme (%d) at %s' % (thisExt, str(thisDate)))
                                if saveLogs:
                                    saveLogs.write('\nDeleting extreme (%d) at %s' % (thisExt, str(thisDate)))
                                dataInd.loc[thisDate] = 0
                        else:
                            lastDate = thisDate
                        try:
                            thisDate = dataInd[dataInd == thisExt][thisDate:][1:].first_valid_index()
                        except IndexError:
                            break
            if showPlots or savePlots:
                plotIndicator(df, dataInd, showPlots=showPlots, savePlots=savePlots, nameSuffix=nameSuffix)
            if saveLogs:
                saveLogs.flush()
            return (dataInd)

        def checkPhaseLength(df, indicator, keepFirst=False, phaseLength=5, meanVal=100, printDetails=True, showPlots=True, savePlots=None, nameSuffix='', saveLogs=None):
            dataInd = indicator.copy()
            if printDetails:
                print('\nChecking extremes at %s for phase length:' % (dataInd.columns[0]))
            if saveLogs:
                saveLogs.write('\nChecking extremes at %s for phase length:' % (dataInd.columns[0]))
            if (dataInd[dataInd != 0].notnull().sum()[0] > 1):  # more than 1 phase?
                lastDate = dataInd[dataInd != 0].first_valid_index()
                lastExt = dataInd.loc[lastDate][0]
                thisDate = dataInd[dataInd != 0][lastDate:][1:].first_valid_index()
                while thisDate:
                    thisExt = dataInd.loc[thisDate][0]
                    realLength = dataInd[lastDate:thisDate].shape[0]
                    if (realLength <= (phaseLength + 1)):  # too short to be a phase?
                        if keepFirst:
                            lastVal = 100  # values are not important for keepFirst option...
                            thisVal = 100  # ... and the df series doesn't have to cover all possible dates
                        else:
                            lastVal = (df.loc[lastDate][0] - meanVal) * lastExt
                            thisVal = (df.loc[thisDate][0] - meanVal) * thisExt
                        if (not (keepFirst) and (thisVal > lastVal)):  # keep the one, which is deviated more (or the earlier one when they equal)
                            if printDetails:
                                print('Deleting extreme (%d) at %s' % (lastExt, str(lastDate)))
                            if saveLogs:
                                saveLogs.write('\nDeleting extreme (%d) at %s' % (lastExt, str(lastDate)))
                            dataInd.loc[lastDate] = 0
                            lastDate = thisDate
                        else:
                            if printDetails:
                                print('Deleting extreme (%d) at %s' % (thisExt, str(thisDate)))
                            if saveLogs:
                                saveLogs.write('\nDeleting extreme (%d) at %s' % (thisExt, str(thisDate)))
                            dataInd.loc[thisDate] = 0
                        if keepFirst:  # the alterations need to be checked after each deletion
                            print('Checking alterations inside the phase length check (because keepFirst parameter is set to True)')
                            dataInd = checkAlterations(df=df, indicator=dataInd, keepFirst=True, showPlots=False)
                            print('\nGetting back to the phase length check:')
                    else:
                        lastDate = thisDate
                        lastExt = thisExt
                    try:
                        thisDate = dataInd[dataInd != 0][thisDate:][1:].first_valid_index()
                    except IndexError:
                        break
            if showPlots or savePlots:
                plotIndicator(df, dataInd, showPlots=showPlots, savePlots=savePlots, nameSuffix=nameSuffix)
            if saveLogs:
                saveLogs.flush()
            return (dataInd)

        col_ind_local = getLocalExtremes(df=col, showPlots=showPlots, savePlots=savePlots, nameSuffix='_04_localExt')
        col_ind_neigh = checkNeighbourhood(df=col, indicator=col_ind_local, printDetails=printDetails,
                                           showPlots=showPlots, saveLogs=saveLogs)
        col_ind_alter = checkAlterations(df=col, indicator=col_ind_neigh, printDetails=printDetails,
                                         showPlots=showPlots, saveLogs=saveLogs)
        col_ind_cycleLength = checkCycleLength(df=col, indicator=col_ind_alter, printDetails=printDetails,
                                               showPlots=showPlots, saveLogs=saveLogs, cycleLength=c)
        col_ind_neighAgain = checkNeighbourhood(df=col, indicator=col_ind_cycleLength, printDetails=printDetails,
                                                showPlots=showPlots, saveLogs=saveLogs)
        col_ind_alterAgain = checkAlterations(df=col, indicator=col_ind_neighAgain, printDetails=printDetails,
                                              showPlots=showPlots, saveLogs=saveLogs)
        col_ind_phaseLength = checkPhaseLength(df=col, indicator=col_ind_alterAgain, printDetails=printDetails,
                                               showPlots=showPlots, saveLogs=saveLogs, phaseLength=h)
        col_ind_neighLast = checkNeighbourhood(df=col, indicator=col_ind_phaseLength, printDetails=printDetails,
                                               showPlots=showPlots, saveLogs=saveLogs)
        col_ind_turningPoints = checkAlterations(df=col, indicator=col_ind_neighLast, printDetails=printDetails,
                                                 showPlots=showPlots, savePlots=savePlots, nameSuffix='_05_ext',
                                                 saveLogs=saveLogs)
        if createInverse:
            colName = col.columns[0]
            col_inv_ind_turningPoints = col_ind_turningPoints.copy() * -1
            col_inv_ind_turningPoints = col_inv_ind_turningPoints.rename(columns={colName: str(colName) + '_INV'})
            return (col_ind_turningPoints, col_inv_ind_turningPoints)
        else:
            return (col_ind_turningPoints)

    indi = pipelineOneColumnTPDetection(df, printDetails=False, showPlots=False, h=h, c=c)
    if showPlots:
        plotIndicator(df, indi, showPlots=True)
    return indi


def matchTurningPoints(ind1, ind2, lagFrom=-9, lagTo=24, printDetails=True, saveLogs=None):
    if (lagTo <= lagFrom):
        if printDetails:
            print("Error: parameter lagTo should be higher than parameter lagFrom.")
        if saveLogs:
            saveLogs.write("\nError: parameter lagTo should be higher than parameter lagFrom.")
            saveLogs.flush()
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    refName = ind1.columns[0]
    indName = ind2.columns[0]
    dataExt = ind1.loc[ind1[refName] != 0].copy()
    dataExt['extOrd'] = range(len(dataExt))
    refIndexes = ind2.index[(ind2.index).isin(ind1.index)]  # same time period as reference series
    dataInd = ind2.loc[refIndexes].copy()
    if (len(refIndexes) == 0):
        if printDetails:
            print("Warning: There is no overlapping period in the time series.")
        if saveLogs:
            saveLogs.write("\nWarning: There is no overlapping period in the time series.")
        refIndexes = ind1.index
        dataInd = pd.DataFrame(0, columns=[indName], index=refIndexes)
    dataInd['extOrd'] = np.nan
    dataInd['time'] = np.nan
    dataInd['missing'] = np.nan  # only turning points that could be in this time series (from the beginning of this series to the end of the reference series)
    dataInd['missingEarly'] = np.nan  # turning points that occured in the reference series before this time series started
    dataInd['extra'] = np.nan
    if len(dataExt) > 0:
        # 因我的数据是月末值，所以改了一下这里 原文件 freq = 'MS'
        shiftedIndex = pd.date_range(start=min(ind2.index) + relativedelta(months=lagFrom),
                                     end=max(ind2.index) + relativedelta(months=lagTo), freq='M')
        ind2Shifted = pd.DataFrame(index=shiftedIndex, data=ind2)
        dataShifted = pd.DataFrame(index=shiftedIndex)
        for i in range(lagFrom, lagTo):  # plus means lead, minus means lag
            dataShifted = pd.concat([dataShifted, ind2Shifted.shift(i).rename(columns={indName: 'shift_' + str(i)})], axis=1)
        for date, row in dataExt.iterrows():
            # date, row = list(dataExt.iterrows())[19]
            thisExt = row.loc[refName]
            thisExtOrd = row.loc['extOrd']
            try:
                dataShiftedThisDate = dataShifted.loc[date]
                possibleExt = pd.DataFrame(dataShiftedThisDate[dataShiftedThisDate == thisExt])
                if (len(possibleExt) == 0):
                    if (date >= min(refIndexes)):  # this turning point could be in the series
                        dataInd.loc[date, ['missing']] = True
                    else:
                        if printDetails:
                            print("Warning: Missing cycle caused by short series (early turning point).")
                        if saveLogs:
                            saveLogs.write("\nWarning: Missing cycle caused by short series (early turning point).")
                        dataInd.loc[date, ['missingEarly']] = True
                else:
                    shifts = [int(i[6:]) for i, j in possibleExt.iterrows()]
                    minShift = min(shifts, key=abs)
                    dateShift = date - relativedelta(months=minShift) - relativedelta(days=7)
                    # 同上，把dateShift变成月末
                    dateShift = pd.date_range(dateShift, periods=1, freq='M')[0]
                    existingOrd = dataInd.loc[dateShift, 'extOrd']
                    if (not (np.isnan(existingOrd))):  # peak/trough is already occupied
                        existingTime = dataInd.loc[dateShift, 'time']
                        if (abs(existingTime) > abs(minShift)):  # new peak/trough is closer
                            if printDetails:
                                print("Warning: Turning point at %s already matched, changing now from order %d to %d." % (
                                    dateShift.strftime("%Y-%m-%d"), existingOrd, thisExtOrd))
                            if saveLogs:
                                saveLogs.write(
                                    "\nWarning: Turning point at %s already matched, changing now from order %d to %d." % (
                                    dateShift.strftime("%Y-%m-%d"), existingOrd, thisExtOrd))
                            existingOrdDate = dataExt[dataExt['extOrd'] == existingOrd].index
                            dataInd.loc[existingOrdDate, ['missing']] = True
                            dataInd.loc[dateShift, 'time'] = minShift
                            dataInd.loc[dateShift, 'extOrd'] = thisExtOrd
                        else:  # new peak/trough is further then the existing one
                            dataInd.loc[date, ['missing']] = True
                    else:  # empty spot
                        dataInd.loc[dateShift, 'time'] = minShift
                        dataInd.loc[dateShift, 'extOrd'] = thisExtOrd
            except KeyError as e:
                print(e)
                if (date >= min(refIndexes)):  # this turning point could be in the series
                    if printDetails:
                        print("Warning: Missing cycle caused by short series (regular turning point).")
                    if saveLogs:
                        saveLogs.write("\nWarning: Missing cycle caused by short series (regular turning point).")
                    dataInd.loc[date, ['missing']] = True
                else:
                    if printDetails:
                        print("Warning: Missing cycle caused by short series (early turning point).")
                    if saveLogs:
                        saveLogs.write("\nWarning: Missing cycle caused by short series (early turning point).")
                    dataInd.loc[date, ['missingEarly']] = True
    else:
        if printDetails:
            print("Warning: There are no turning points in the reference series.")
        if saveLogs:
            saveLogs.write("\nWarning: There are no turning points in the reference series.")
    dataInd.sort_index(inplace=True)
    lastOrder = 0
    lastTime = None
    lastDate = None
    for thisDate, row in dataInd[dataInd['extOrd'].notnull()].iterrows():
        thisOrder = row['extOrd']
        thisTime = row['time']
        if (thisOrder < lastOrder):
            if printDetails:
                print("Warning: Discrepancy between order of turning points %s and %s." % (
                lastDate.strftime("%Y-%m-%d"), thisDate.strftime("%Y-%m-%d")))
            if saveLogs:
                saveLogs.write("\nWarning: Discrepancy between order of turning points %s and %s." % (
                lastDate.strftime("%Y-%m-%d"), thisDate.strftime("%Y-%m-%d")))
            if (abs(thisTime) < abs(lastTime)):  # keep the one which is closer to the turning point
                if printDetails:
                    print("<-- %s deleted from matched turning points." % lastDate.strftime("%Y-%m-%d"))
                if saveLogs:
                    saveLogs.write("\n<-- %s deleted from matched turning points." % lastDate.strftime("%Y-%m-%d"))
                dataInd.loc[lastDate, 'extOrd'] = np.nan
                dataInd.loc[lastDate, 'time'] = np.nan
                lastOrdDate = dataExt[dataExt['extOrd'] == lastOrder].index
                dataInd.loc[lastOrdDate[0], ['missing']] = True
                lastOrder = thisOrder
                lastTime = thisTime
                lastDate = thisDate
            else:
                if printDetails:
                    print("<-- %s deleted from matched turning points." % thisDate.strftime("%Y-%m-%d"))
                if saveLogs:
                    saveLogs.write("\n<-- %s deleted from matched turning points." % thisDate.strftime("%Y-%m-%d"))
                dataInd.loc[thisDate, 'extOrd'] = np.nan
                dataInd.loc[thisDate, 'time'] = np.nan
                thisOrdDate = dataExt[dataExt['extOrd'] == thisOrder].index
                dataInd.loc[thisOrdDate[0], ['missing']] = True
        else:
            lastOrder = thisOrder
            lastTime = thisTime
            lastDate = thisDate
    dataInd.loc[((dataInd[indName] != 0)
                 & (dataInd[indName].notnull())
                 & (dataInd['extOrd'].isnull())), 'extra'] = True
    lastExt = dataInd['extOrd'].last_valid_index()
    lastExtra = dataInd['extra'].last_valid_index()
    if (
            (
                    lastExtra != None
                    and
                    #  原代码没有减7天
                    (lastExtra > (ind1.last_valid_index() - relativedelta(months=lagTo) - relativedelta(days=7)))
            )
            and
            (
                    lastExt == None
                    or
                    (lastExtra > lastExt)
            )
    ):
        if printDetails:
            print("Warning: Last extreme wasn\'t marked as extra, because it was too close to the end of reference series.")
        if saveLogs:
            saveLogs.write("\nWarning: Last extreme wasn\'t marked as extra, because it was too close to the end of reference series.")
        dataInd.loc[lastExtra, 'extra'] = np.nan
    if saveLogs:
        saveLogs.flush()
    # Return results
    return (pd.DataFrame(dataInd['extOrd']).rename(columns={'extOrd': indName})
            , pd.DataFrame(dataInd['time']).rename(columns={'time': indName})
            , pd.DataFrame(dataInd['missing']).rename(columns={'missing': indName})
            , pd.DataFrame(dataInd['missingEarly']).rename(columns={'missingEarly': indName})
            , pd.DataFrame(dataInd['extra']).rename(columns={'extra': indName}))
